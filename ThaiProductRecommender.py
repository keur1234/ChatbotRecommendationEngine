
from pythainlp.corpus.common import thai_stopwords
from pythainlp.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
thai_stop_words = thai_stopwords()

class ThaiProductRecommender:
  
  """
  This class recommends products based on user input and product descriptions.
  """
  def __init__(self, df):
    """
    Initialize the class with a pandas dataframe containing product data and Thai stop words.

    Args:
      df (pandas.DataFrame): Dataframe containing product information.
      thai_stop_words (list): List of Thai stop words to remove during tokenization.
    """
    self.df = df
    self.thai_stop_words = thai_stopwords()


    # Feature extraction (performed in __init__ for efficiency)
    self.tfidf_vectorizer = TfidfVectorizer(tokenizer=self.thai_tokenizer)
    self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(df['name'] + ' ' + df['description'])
    self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)

  def thai_tokenizer(self, text):
    tokens = word_tokenize(text, engine='newmm')
    return [word for word in tokens if word not in self.thai_stop_words]

  def get_recommendations(self, user_input, category=None, brand=None, price_range=None):
    """
    Recommends products based on user input and optional filtering criteria.

    Args:
      user_input (str): User input describing the desired product.
      category (str, optional): Category to filter recommendations (default: None).
      brand (str, optional): Brand to filter recommendations (default: None).
      price_range (tuple, optional): Min and max price for recommendations (default: None).

    Returns:
      pandas.DataFrame: Dataframe containing recommended products or "ไม่พบ" if none found.
    """
    user_input_vector = self.tfidf_vectorizer.transform([user_input])
    sim_scores = linear_kernel(user_input_vector, self.tfidf_matrix).flatten()
    sim_scores = list(enumerate(sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:5]  # Get top 4 recommendations
    product_indices = [i[0] for i in sim_scores]

    recommended_products = self.df[['product_id', 'name', 'category', 'price', 'brand', 'warranty_period', 'stock_quantity']].iloc[product_indices]

    if category:
      recommended_products = recommended_products[recommended_products['category'] == category]

    if brand:
      recommended_products = recommended_products[recommended_products['brand'] == brand]

    if price_range:
      min_price, max_price = price_range
      recommended_products = recommended_products[(recommended_products['price'] >= min_price) & (recommended_products['price'] <= max_price)]

    return recommended_products if not recommended_products.empty else "ไม่มีสินค้าที่ตรงกับรายการ"