import json
import os
import glob
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Document:
    id: str
    content: str
    metadata: Dict
    embedding: Optional[List[float]] = None


class DocumentLoader:
    def __init__(self, data_directory: str = "RAG"):
        self.data_directory = data_directory
        self.required_fields = ["id", "content"]

    def load_documents(self) -> List[Document]:
        documents = []
        json_files = glob.glob(f"{self.data_directory}/*.json")
        
        if not json_files:
            logger.warning(f"Нет JSON файлов в {self.data_directory}")
            return documents
            
        logger.info(f"Найдено {len(json_files)} документов")
        
        for file_path in json_files:
            doc = self._load_single_document(file_path)
            if doc:
                documents.append(doc)
        
        logger.info(f"Загружено {len(documents)} документов")
        return documents

    def _load_single_document(self, file_path: str) -> Optional[Document]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not self._validate_document(data):
                logger.warning(f"Документ {file_path} не прошел валидацию")
                return None
            
            # Собираем metadata (всё кроме id и content)
            metadata = {k: v for k, v in data.items() if k not in ['id', 'content']}
            metadata['source_file'] = os.path.basename(file_path)

            return Document(
                id=data['id'],
                content=data['content'],
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"Ошибка загрузки {file_path}: {e}")
            return None
        

    def _validate_document(self, data: Dict) -> bool:
        for field in self.required_fields:
            if field not in data or not data[field]:
                logger.warning(f"Отсутствует поле: {field}")
                return False
        return True