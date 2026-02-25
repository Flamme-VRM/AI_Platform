import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class HistoricalMetrics:
    """Historical bot usage data"""
    total_unique_users: int = 0
    first_user_date: Optional[str] = None
    total_sessions: int = 0
    avg_session_length: float = 0.0
    user_retention: Dict[str, float] = None
    popular_subjects: List[Dict] = None

    def __post_init__(self):
        if self.user_retention is None:
            self.user_retention = {}
        if self.popular_subjects is None:
            self.popular_subjects = []

class AnalyticsService:
    """Service for historical analytics and persistent metrics"""

    def __init__(self, db_path: str = "bot_analytics.db"):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize analytics database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                -- Users table for historical user data
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    first_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
                    total_messages INTEGER DEFAULT 0,
                    total_voice_messages INTEGER DEFAULT 0,
                    language TEXT DEFAULT 'kk',
                    is_active BOOLEAN DEFAULT 1
                );

                -- User sessions for retention analysis
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    session_start DATETIME DEFAULT CURRENT_TIMESTAMP,
                    session_end DATETIME,
                    messages_count INTEGER DEFAULT 0,
                    subjects_discussed TEXT, -- JSON array
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                );

                -- Subject popularity tracking
                CREATE TABLE IF NOT EXISTS subject_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE,
                    subject TEXT,
                    user_count INTEGER DEFAULT 0,
                    message_count INTEGER DEFAULT 0,
                    UNIQUE(date, subject)
                );

                -- Daily bot statistics
                CREATE TABLE IF NOT EXISTS daily_stats (
                    date DATE PRIMARY KEY,
                    unique_users INTEGER DEFAULT 0,
                    total_messages INTEGER DEFAULT 0,
                    total_voice_messages INTEGER DEFAULT 0,
                    rag_usage_count INTEGER DEFAULT 0,
                    avg_response_time REAL DEFAULT 0.0,
                    top_subjects TEXT -- JSON array of top subjects
                );

                -- Create indexes for performance
                CREATE INDEX IF NOT EXISTS idx_users_last_seen ON users(last_seen);
                CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON user_sessions(user_id);
                CREATE INDEX IF NOT EXISTS idx_sessions_start ON user_sessions(session_start);
                CREATE INDEX IF NOT EXISTS idx_subject_usage_date ON subject_usage(date);
            """)

        logger.info(f"Analytics database initialized: {self.db_path}")

    def record_user_activity(self, user_id: int, message_type: str = "text",
                           subject: str = None, response_time: float = None):
        """Record user activity for historical analysis"""
        with sqlite3.connect(self.db_path) as conn:
            # Upsert user record
            conn.execute("""
                INSERT OR REPLACE INTO users
                (user_id, last_seen, total_messages, total_voice_messages)
                VALUES (?, CURRENT_TIMESTAMP,
                    COALESCE((SELECT total_messages FROM users WHERE user_id = ?), 0) + 1,
                    COALESCE((SELECT total_voice_messages FROM users WHERE user_id = ?), 0)
                    + CASE WHEN ? = 'voice' THEN 1 ELSE 0 END
                )
            """, (user_id, user_id, user_id, message_type))

            # Record daily stats
            today = datetime.now().strftime("%Y-%m-%d")
            conn.execute("""
                INSERT OR REPLACE INTO daily_stats
                (date, unique_users, total_messages, total_voice_messages)
                VALUES (?, COALESCE(
                    (SELECT unique_users FROM daily_stats WHERE date = ?), 0
                ), COALESCE(
                    (SELECT total_messages FROM daily_stats WHERE date = ?), 0
                ) + 1, COALESCE(
                    (SELECT total_voice_messages FROM daily_stats WHERE date = ?), 0
                ) + CASE WHEN ? = 'voice' THEN 1 ELSE 0 END
                )
            """, (today, today, today, today, message_type))

            # Record subject usage
            if subject:
                conn.execute("""
                    INSERT OR REPLACE INTO subject_usage
                    (date, subject, user_count, message_count)
                    VALUES (?, ?, 1, 1)
                """, (today, subject))

        logger.debug(f"Recorded activity for user {user_id}")

    def get_total_unique_users(self) -> int:
        """Get total number of unique users ever"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM users")
            return cursor.fetchone()[0]

    def get_user_retention_data(self) -> Dict[str, float]:
        """Calculate user retention rates"""
        with sqlite3.connect(self.db_path) as conn:
            # Get users by registration period
            retention_data = {}

            # Daily active users for last 30 days
            thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

            cursor = conn.execute("""
                SELECT DATE(last_seen) as activity_date,
                       COUNT(*) as active_users
                FROM users
                WHERE last_seen >= ?
                GROUP BY DATE(last_seen)
                ORDER BY activity_date
            """, (thirty_days_ago,))

            daily_activity = cursor.fetchall()

            # Calculate retention (users active in last 7 days vs total)
            seven_days_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

            cursor = conn.execute("""
                SELECT COUNT(*) FROM users WHERE last_seen >= ?
            """, (seven_days_ago,))

            recent_users = cursor.fetchone()[0]
            total_users = self.get_total_unique_users()

            retention_data['7_day_retention'] = round(
                (recent_users / max(total_users, 1)) * 100, 2
            )

            return retention_data

    def get_popular_subjects(self, days: int = 30) -> List[Dict]:
        """Get most popular subjects in last N days"""
        with sqlite3.connect(self.db_path) as conn:
            cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

            cursor = conn.execute("""
                SELECT subject, SUM(message_count) as total_messages
                FROM subject_usage
                WHERE date >= ?
                GROUP BY subject
                ORDER BY total_messages DESC
                LIMIT 10
            """, (cutoff_date,))

            results = cursor.fetchall()
            return [
                {"subject": subject, "message_count": count}
                for subject, count in results
            ]

    def get_usage_trends(self, days: int = 30) -> Dict:
        """Get usage trends over time"""
        with sqlite3.connect(self.db_path) as conn:
            cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

            cursor = conn.execute("""
                SELECT date, unique_users, total_messages, avg_response_time
                FROM daily_stats
                WHERE date >= ?
                ORDER BY date
            """, (cutoff_date,))

            trends = cursor.fetchall()
            return {
                "days": days,
                "data": [
                    {
                        "date": date,
                        "unique_users": users,
                        "total_messages": messages,
                        "avg_response_time": response_time
                    }
                    for date, users, messages, response_time in trends
                ]
            }

    def get_comprehensive_analytics(self) -> Dict:
        """Get comprehensive analytics report"""
        return {
            "overview": {
                "total_unique_users": self.get_total_unique_users(),
                "user_retention": self.get_user_retention_data(),
                "database_size": "N/A"  # Could calculate actual DB size
            },
            "usage_patterns": {
                "popular_subjects": self.get_popular_subjects(),
                "usage_trends": self.get_usage_trends()
            },
            "performance": {
                "avg_response_time": "N/A",  # Would need to store this
                "peak_usage_days": "N/A"  # Could calculate from trends
            }
        }

    def export_analytics_report(self, filename: str = None) -> str:
        """Export analytics to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bot_analytics_{timestamp}.json"

        analytics_data = self.get_comprehensive_analytics()

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(analytics_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Analytics exported to {filename}")
        return filename