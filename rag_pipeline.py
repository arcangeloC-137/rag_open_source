from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from typing import List
from opensearchpy import OpenSearch
import os
import json

class RAGPipeline:
    def __init__(self, opensearch_url: str, username: str, password: str, index_name: str = "rag-index"):
        self.embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")
        self.index_name = index_name
        
        # Inizializza il client OpenSearch
        self.client = OpenSearch(
            hosts=[opensearch_url],
            http_auth=(username, password),
            use_ssl=True,
            verify_certs=False
        )
        
        # Crea l'indice se non esiste
        self._create_index_if_not_exists()
        
        # Inizializza vectorstore dopo aver creato l'indice
        self.vectorstore = OpenSearchVectorSearch(
            opensearch_url=opensearch_url,
            index_name=index_name,
            embedding_function=self.embedding_model,
            http_auth=(username, password),
            use_ssl=True,
            verify_certs=False
        )

    def process_pdf(self, file_path: str) -> List[Document]:
        """Carica un PDF e lo converte in una lista di documenti"""
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        docs = []
        file_name = os.path.basename(file_path)

        print(f"NOME DEL FILE: [{file_name}]")
        for i, page in enumerate(pages):
            doc = Document(
                page_content=page.page_content,
                metadata={"source": file_name, "page": i + 1}
            )
            docs.append(doc)

        return docs

    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        """Suddivide i documenti in chunks più piccoli"""
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        return splitter.split_documents(docs)

    def index_documents(self, chunks: List[Document]):
        """Indicizza i chunks nel vectorstore"""
        print(f"Indexing {len(chunks)} chunks...")
        if chunks:
            print(f"Example chunk: {chunks[0].page_content[:100]}...")
            print(f"Example metadata: {chunks[0].metadata}")
        
        # Aggiungi i documenti al vectorstore
        self.vectorstore.add_documents(chunks)
        print("Indexing complete")

    def run_pipeline(self, file_path: str):
        """Esegue l'intera pipeline di indicizzazione"""
        docs = self.process_pdf(file_path)
        chunks = self.chunk_documents(docs)
        self.index_documents(chunks)
    
    def delete_document(self, document_name: str):
        """Elimina un documento dall'indice usando il nome del file"""
        self.client.delete_by_query(
            index=self.index_name,
            body={
                "query": {
                    "term": {
                        "metadata.source.keyword": document_name
                    }
                }
            }
        )
    
    def flush_all(self):
        """Elimina tutti i documenti e ricrea l'indice"""
        try:
            self.client.indices.delete(index=self.index_name, ignore=[400, 404])
            self._create_index_if_not_exists()
            print("Index deleted and recreated")
        except Exception as e:
            print(f"Error flushing index: {e}")
        
        # Elimina anche i file locali
        try:
            documents_dir = "./documents"
            for filename in os.listdir(documents_dir):
                file_path = os.path.join(documents_dir, filename)
                if os.path.isfile(file_path) and not filename.startswith('.'):
                    os.remove(file_path)
            print("Local files deleted")
        except Exception as e:
            print(f"Error deleting local files: {e}")

    def get_indexed_documents(self):
        """Restituisce un elenco di nomi di documenti indicizzati"""
        try:
            # Prima verifica la struttura esaminando un documento
            sample = self.client.search(
                index=self.index_name,
                body={
                    "size": 1,
                    "query": {"match_all": {}}
                }
            )
            
            # Per sicurezza, stampa la struttura del primo documento
            if 'hits' in sample and 'hits' in sample['hits'] and len(sample['hits']['hits']) > 0:
                first_doc = sample['hits']['hits'][0]['_source']
                print("Document structure:", json.dumps(first_doc, indent=2))
                
                # Estrai tutti i valori "source" unici
                response = self.client.search(
                    index=self.index_name,
                    body={
                        "size": 0,
                        "aggs": {
                            "sources": {
                                "terms": {
                                    # Prova diversi percorsi del campo in base alla struttura reale
                                    "field": "metadata.source.keyword", 
                                    "size": 100
                                }
                            }
                        }
                    }
                )
                
                # Se l'aggregazione non funziona, prova un'altra soluzione
                if ('aggregations' in response and 
                    'sources' in response['aggregations'] and 
                    len(response['aggregations']['sources']['buckets']) > 0):
                    # L'aggregazione ha funzionato
                    return [bucket["key"] for bucket in response["aggregations"]["sources"]["buckets"]]
                else:
                    # Se l'aggregazione standard non funziona, recupera tutti i documenti e estrai i nomi dei file
                    print("Aggregation failed, switching to full scan...")
                    scan_response = self.client.search(
                        index=self.index_name,
                        body={
                            "size": 1000,  # Prendi i primi 1000 documenti
                            "_source": ["metadata.source", "metadata"],  # Carica solo i metadati
                            "query": {"match_all": {}}
                        }
                    )
                    
                    # Estrai i nomi dei file dai documenti
                    source_set = set()
                    for hit in scan_response['hits']['hits']:
                        source = hit['_source']
                        if 'metadata' in source and 'source' in source['metadata']:
                            source_set.add(source['metadata']['source'])
                    
                    print(f"Found {len(source_set)} unique sources: {source_set}")
                    return list(source_set)
            return []
        except Exception as e:
            print(f"Error getting indexed documents: {str(e)}")
            return []
    
    def _create_index_if_not_exists(self):
        """Crea l'indice se non esiste"""
        if not self.client.indices.exists(index=self.index_name):
            # Dimensione del vettore per il modello e5-base-v2
            embedding_dim = 768
            
            mapping = {
                "settings": {
                    "index": {
                        "knn": True,
                        "knn.space_type": "cosinesimil"
                    }
                },
                "mappings": {
                    "properties": {
                        "vector_field": {  # Campo per i vettori
                            "type": "knn_vector",
                            "dimension": embedding_dim
                        },
                        "text": {  # Campo per il contenuto
                            "type": "text"
                        },
                        "metadata": {  # Campo per i metadati
                            "properties": {
                                "source": {"type": "keyword"},
                                "page": {"type": "integer"}
                            }
                        }
                    }
                }
            }
            
            print(f"Creating index {self.index_name} with mapping")
            self.client.indices.create(index=self.index_name, body=mapping)
