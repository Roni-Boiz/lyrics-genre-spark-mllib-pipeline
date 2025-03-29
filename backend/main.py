import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import random

from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml import Transformer
from pyspark.ml.util import  MLReadable, MLWritable, DefaultParamsWriter, DefaultParamsReader
from pyspark.ml.feature import Tokenizer, RegexTokenizer, StopWordsRemover, Word2Vec, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.param.shared import Param, Params
from pyspark.sql.types import StringType, ArrayType, IntegerType
from pyspark.sql.functions import regexp_replace, monotonically_increasing_id, explode, col, udf, lower, concat, concat_ws, array_distinct, collect_list, expr
import re

app = FastAPI()

spark_session = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
    
spark_session = SparkSession.builder.master("local[*]") \
                                    .appName("FastAPILyricsApp") \
                                    .getOrCreate()

GENRES = ["Pop", "Country", "Blues", "Rock", "Jazz", "Raggae", "Hip-Hop", "Classical Soul"]

class LyricsRequest(BaseModel):
    lyrics: str

@app.post("/predict")
def predict_genre(request: LyricsRequest):
    # Get the lyrics from the request
    lyrics_text = request.lyrics
    print(lyrics_text)

    if not lyrics_text.strip():
        raise HTTPException(status_code=400, detail="Lyrics cannot be empty")
    
    loaded_model = PipelineModel.load("./../model")
    lyrics_df = spark_session.createDataFrame([(lyrics_text,)], ["lyrics"])

    predictions = loaded_model.transform(lyrics_df)
    probabilities = predictions.select("probability").collect()[0]["probability"]
    print(probabilities)

    # Generate random scores between 0 and 1 for each genre
    scores = {GENRES[i]: round(probabilities[i], 2) for i in range(len(GENRES))}
    print(scores)

    return {"predictions": scores}

class Cleanser(Transformer, MLReadable, MLWritable):
    def _transform(self, dataset):
        return dataset.withColumn("cleaned_lyrics", regexp_replace(lower(col("lyrics")), "[^a-zA-Z\\s]", ""))
    
    def write(self):
        """Returns an MLWriter instance for saving this ML instance."""
        return DefaultParamsWriter(self) # Use DefaultParamsWriter

    def saveImpl(self, path):
        """Saves the parameters to the input path."""
        DefaultParamsWriter.saveImpl(self, path) # Use DefaultParamsWriter

    @classmethod
    def read(cls):
        """Returns an MLReader instance for loading this ML instance."""
        return DefaultParamsReader(cls)

    @classmethod
    def load(cls, path):
        """Loads the ML instance from the input path."""
        reader = DefaultParamsReader(cls)
        return reader.load(path)

class Numerator(Transformer, MLReadable, MLWritable):
    def _transform(self, dataset):
        return dataset.withColumn("line_number", monotonically_increasing_id())
    
    def write(self):
        """Returns an MLWriter instance for saving this ML instance."""
        return DefaultParamsWriter(self) # Use DefaultParamsWriter

    def saveImpl(self, path):
        """Saves the parameters to the input path."""
        DefaultParamsWriter.saveImpl(self, path) # Use DefaultParamsWriter

    @classmethod
    def read(cls):
        """Returns an MLReader instance for loading this ML instance."""
        return DefaultParamsReader(cls)

    @classmethod
    def load(cls, path):
        """Loads the ML instance from the input path."""
        reader = DefaultParamsReader(cls)
        return reader.load(path)

class Exploder(Transformer, MLReadable, MLWritable):
    def _transform(self, dataset):
        return dataset.withColumn("words", explode(col("filtered_tokens")))
    
    def write(self):
        """Returns an MLWriter instance for saving this ML instance."""
        return DefaultParamsWriter(self) # Use DefaultParamsWriter

    def saveImpl(self, path):
        """Saves the parameters to the input path."""
        DefaultParamsWriter.saveImpl(self, path) # Use DefaultParamsWriter

    @classmethod
    def read(cls):
        """Returns an MLReader instance for loading this ML instance."""
        return DefaultParamsReader(cls)

    @classmethod
    def load(cls, path):
        """Loads the ML instance from the input path."""
        reader = DefaultParamsReader(cls)
        return reader.load(path)

class Stemmer(Transformer, MLReadable, MLWritable):
    def _transform(self, dataset):
        remove_suffix_udf = udf(lambda word: re.sub(r'(ing|ed|ly|es|s)$', '', word), StringType())
        return dataset.withColumn("stemmed_word", remove_suffix_udf(col("words")))

    def write(self):
        """Returns an MLWriter instance for saving this ML instance."""
        return DefaultParamsWriter(self) # Use DefaultParamsWriter

    def saveImpl(self, path):
        """Saves the parameters to the input path."""
        DefaultParamsWriter.saveImpl(self, path) # Use DefaultParamsWriter
    
    @classmethod
    def read(cls):
        """Returns an MLReader instance for loading this ML instance."""
        return DefaultParamsReader(cls)

    @classmethod
    def load(cls, path):
        """Loads the ML instance from the input path."""
        reader = DefaultParamsReader(cls)
        return reader.load(path)

class Uniter(Transformer, MLReadable, MLWritable):
    def _transform(self, dataset):
        const_lyrics = dataset.groupBy("line_number").agg(concat_ws(" ", collect_list("stemmed_word")).alias("reconstructed_lyrics")).orderBy("line_number")
        return dataset.drop("words", "stemmed_word").dropDuplicates().orderBy("line_number").join(const_lyrics, "line_number", "right").orderBy("line_number")

    def write(self):
        """Returns an MLWriter instance for saving this ML instance."""
        return DefaultParamsWriter(self) # Use DefaultParamsWriter

    def saveImpl(self, path):
        """Saves the parameters to the input path."""
        DefaultParamsWriter.saveImpl(self, path) # Use DefaultParamsWriter
        
    @classmethod
    def read(cls):
        """Returns an MLReader instance for loading this ML instance."""
        return DefaultParamsReader(cls)

    @classmethod
    def load(cls, path):
        """Loads the ML instance from the input path."""
        reader = DefaultParamsReader(cls)
        return reader.load(path)

class Verser(Transformer, MLReadable, MLWritable):
    def __init__(self, sentences_in_verse=16):
        super().__init__()
        self.sentences_in_verse = sentences_in_verse

    def _transform(self, dataset):
        def create_verses(lyrics):
            words = lyrics.split()
            verses = [" ".join(words[i:i + self.sentences_in_verse]) for i in range(0, len(words), self.sentences_in_verse)]
            return verses

        verse_udf = udf(create_verses, ArrayType(StringType()))

        return dataset.withColumn("verses", verse_udf(dataset["reconstructed_lyrics"]))
    
    def write(self):
        """Returns an MLWriter instance for saving this ML instance."""
        return DefaultParamsWriter(self) # Use DefaultParamsWriter

    def saveImpl(self, path):
        """Saves the parameters to the input path."""
        DefaultParamsWriter.saveImpl(self, path) # Use DefaultParamsWriter

    @classmethod
    def read(cls):
        """Returns an MLReader instance for loading this ML instance."""
        return DefaultParamsReader(cls)

    @classmethod
    def load(cls, path):
        """Loads the ML instance from the input path."""
        reader = DefaultParamsReader(cls)
        return reader.load(path)

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)

