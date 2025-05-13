from langchain.vectorstores import OpenSearchVectorSearch
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
from opensearchpy import OpenSearch
import os
import json

class RAG:
    def __init__(self, opensearch_url: str, username: str, password: str, index_name: str = "rag-index"):
        embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")
        self.vectorstore = OpenSearchVectorSearch(
            opensearch_url=opensearch_url,
            embedding_function=embedding_model,
            index_name=index_name,
            http_auth=(username, password),
            use_ssl=True,
            verify_certs=False
        )

        self.client = OpenSearch(
            hosts=[opensearch_url],
            http_auth=(username, password),
            use_ssl=True,
            verify_certs=False
        )
    
    def retrieve_chunks(self, query: str, document_name: str, k=5):
        """Recupera i chunks più rilevanti per la query dal documento specificato"""
        print(f"Cercando chunks per document_name: {document_name}")
        
        # Verifica prima la struttura esatta
        check_query = {
            "size": 5,
            "query": {
                "match_all": {}
            }
        }
        
        check_result = self.client.search(
            index=self.vectorstore.index_name,
            body=check_query
        )
        
        if 'hits' in check_result and 'hits' in check_result['hits'] and len(check_result['hits']['hits']) > 0:
            sample_doc = check_result['hits']['hits'][0]['_source']
            print(f"Struttura documento di esempio: {json.dumps(list(sample_doc.keys()), indent=2)}")
            if 'metadata' in sample_doc:
                print(f"Struttura metadata: {json.dumps(dict(sample_doc['metadata']), indent=2)}")
        
        # Prova diverse varianti del filtro
        filters_to_try = [
            {"term": {"metadata.source": document_name}},
            {"term": {"metadata.source.keyword": document_name}},
            {"wildcard": {"metadata.source": f"*{document_name}*"}}
        ]
        
        results = None
        for filter_option in filters_to_try:
            print(f"Tentativo con filtro: {filter_option}")
            try:
                results = self.vectorstore.similarity_search(
                    query,
                    k=k,
                    filter=filter_option
                )
                
                if results and len(results) > 0:
                    print(f"Filtro funzionante: {filter_option}")
                    break
            except Exception as e:
                print(f"Errore con filtro {filter_option}: {e}")
        
        # Se ancora non abbiamo risultati, proviamo un approccio diretto con client.search
        if not results or len(results) == 0:
            print("Tentativo diretto con client.search...")
            try:
                # Ottieni l'embedding della query
                query_embedding = self.vectorstore.embedding_function.embed_query(query)
                
                # Costruisci una query di ricerca vettoriale con filtro
                vector_field = "vector_field" if "vector_field" in sample_doc else "vector"
                
                search_query = {
                    "size": k,
                    "query": {
                        "script_score": {
                            "query": {
                                "bool": {
                                    "must": [
                                        {"term": {"metadata.source": document_name}}
                                    ]
                                }
                            },
                            "script": {
                                "source": f"cosineSimilarity(params.query_vector, '{vector_field}') + 1.0",
                                "params": {"query_vector": query_embedding}
                            }
                        }
                    }
                }
                
                response = self.client.search(
                    index=self.vectorstore.index_name,
                    body=search_query
                )
                
                # Converti i risultati nel formato Document di LangChain
                from langchain_core.documents import Document
                results = []
                for hit in response['hits']['hits']:
                    source = hit["_source"]
                    doc = Document(
                        page_content=source.get("text", ""),
                        metadata=source.get("metadata", {})
                    )
                    results.append(doc)
            except Exception as e:
                print(f"Errore nell'approccio diretto: {e}")
        
        print(f"Trovati {len(results) if results else 0} chunks")
        if results and len(results) > 0:
            print(f"Primo chunk: {results[0].page_content[:50]}...")
            print(f"Metadata del primo chunk: {results[0].metadata}")
        else:
            print("Nessun chunk trovato!")
        
        return results or []
    
    def construct_prompt(self, query: str, documents: List, prompt_path: str = "default.txt") -> str:
        """Costruisce il prompt per il modello utilizzando i chunks recuperati"""
        if not documents:
            return f"Non ho trovato informazioni pertinenti. Prova a rispondere alla domanda: {query}"
        
        print(f"DOC METADATA: {documents[0]}")
        # documents = documents.sort(key=lambda x: x.metadata['page'])
        context = "\n\n".join([f"Doc name: {doc.metadata['source']}, Page: {doc.metadata['page']}\n{doc.page_content}" for doc in documents])
        print(f"Contesto costruito con {len(documents)} chunks")
        
        with open(f"prompts/{prompt_path}") as prompt_doc:
            prompt = prompt_doc.read()

        return f"""
{prompt}

CONTEXT:
{context}

QUESTION:
{query}


"""
    
    def debug_index(self):
        """Funzione di debug per verificare la struttura dell'indice"""
        try:
            # Ottieni i primi 5 documenti
            response = self.vectorstore.client.search(
                index=self.vectorstore.index_name,
                body={
                    "size": 5,
                    "query": {
                        "match_all": {}
                    }
                }
            )
            
            # Stampa la struttura
            print("STRUTTURA INDICE:")
            if 'hits' in response and 'hits' in response['hits']:
                if len(response['hits']['hits']) == 0:
                    print("NESSUN DOCUMENTO TROVATO NELL'INDICE")
                for hit in response['hits']['hits']:
                    print("ID documento:", hit['_id'])
                    print("Struttura source:", json.dumps(hit['_source'], indent=2))
                    print("-" * 50)
            return response
        except Exception as e:
            print(f"Errore in debug_index: {e}")
            return None


import hashlib

def compute_file_hash(file_bytes):
    """Calcola l'hash SHA-256 di un file in formato bytes"""
    return hashlib.sha256(file_bytes).hexdigest()

def inspect_documents(self):
    """Esamina i primi documenti per vedere la loro struttura"""
    try:
        response = self.vectorstore.client.search(
            index=self.vectorstore.index_name,
            body={
                "size": 5,  # prendi solo i primi 5
                "query": {
                    "match_all": {}
                }
            }
        )
        
        if 'hits' in response and 'hits' in response['hits']:
            documents = []
            for hit in response['hits']['hits']:
                documents.append(hit['_source'])
            return documents
        return []
    except Exception as e:
        print(f"Error inspecting documents: {e}")
        return []