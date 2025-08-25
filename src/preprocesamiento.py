# src/preprocesamiento.py
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import spacy

# Descargar recursos de NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class PreprocesadorTexto:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        # Cargar modelo de spaCy para procesamiento avanzado
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            # Si no está instalado, lo instalamos
            import os
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
    
    def limpiar_texto(self, texto):
        if not isinstance(texto, str):
            return ""
        
        # Convertir a minúsculas
        texto = texto.lower()
        
        # Eliminar caracteres especiales y números
        texto = re.sub(r'[^a-zA-Z\s]', '', texto)
        
        # Tokenizar
        tokens = word_tokenize(texto)
        
        # Eliminar stopwords y aplicar stemming
        tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words]
        
        # Unir tokens
        texto_limpio = ' '.join(tokens)
        
        return texto_limpio
    
    def preprocesar_spacy(self, texto):
        """Procesamiento más avanzado con spaCy"""
        if not isinstance(texto, str):
            return ""
        
        doc = self.nlp(texto)
        # Lematización y filtrar stopwords, puntuación
        tokens = [token.lemma_.lower() for token in doc 
                 if not token.is_stop and not token.is_punct and not token.is_space]
        
        return ' '.join(tokens)
    
    def preprocesar_dataframe(self, df, columnas, usar_spacy=False):
        """Preprocesar múltiples columnas de un DataFrame"""
        df_procesado = df.copy()
        
        for columna in columnas:
            print(f"Procesando columna: {columna}")
            if usar_spacy:
                df_procesado[columna] = df_procesado[columna].apply(self.preprocesar_spacy)
            else:
                df_procesado[columna] = df_procesado[columna].apply(self.limpiar_texto)
        
        return df_procesado

# Función para combinar título y abstract
def combinar_textos(df):
    """Combinar título y abstract en un solo texto"""
    df['texto_combinado'] = df['title'] + ' ' + df['abstract']
    return df