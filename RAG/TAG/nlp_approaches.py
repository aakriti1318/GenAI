from openai import OpenAI
import pandas as pd
from typing import Dict, List

# Set up OpenAI API key (replace with your actual key)
client = OpenAI(api_key="...")

# Simulated knowledge base
knowledge_base = {
    "laptop": "A portable computer suitable for use while traveling.",
    "MacBook Pro": "A high-end laptop computer produced by Apple Inc., known for its powerful performance and sleek design.",
    "Dell XPS": "A line of high-performance laptops produced by Dell, featuring premium build quality and compact form factors.",
    "Lenovo ThinkPad": "A series of business-oriented laptops known for their durability and iconic TrackPoint.",
    "HP Spectre": "A line of premium laptops from HP, featuring thin designs and high-end specifications."
}

# Simulated database
products_df = pd.DataFrame({
    'product_id': [1, 2, 3, 4],
    'name': ['MacBook Pro', 'Dell XPS', 'Lenovo ThinkPad', 'HP Spectre'],
    'price': [1999, 1499, 1299, 1399],
    'category': ['laptop', 'laptop', 'laptop', 'laptop'],
    'sales': [1000, 800, 600, 400]
})

def gpt4_call(prompt: str, system_message: str = "You are a helpful assistant.") -> str:
    """Make a call to GPT-4 API."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",  
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def text_to_sql(query: str) -> str:
    """Simulates Text-to-SQL by generating a simple SQL query."""
    prompt = f"""
    Convert the following natural language query to SQL:
    "{query}"
    Assume a table named 'products' with columns: product_id, name, price, category, sales.
    """
    sql_query = gpt4_call(prompt, "You are an SQL expert. Provide only the SQL query without any explanation.")
    return sql_query

def execute_sql(sql_query: str, df: pd.DataFrame) -> pd.DataFrame:
    """Simulates executing SQL query on the DataFrame."""
    # This is a simplified execution. In real-world scenarios, you'd use a proper SQL engine.
    if "ORDER BY sales DESC LIMIT 1" in sql_query:
        return df.sort_values('sales', ascending=False).head(1)
    elif "COUNT" in sql_query:
        return pd.DataFrame([{'count': len(df)}])
    else:
        return df

def rag_with_gpt4(query: str, knowledge_base: Dict[str, str]) -> str:
    """Implements RAG using GPT-4 for generation."""
    relevant_info = [value for key, value in knowledge_base.items() if key.lower() in query.lower()]
    context = " ".join(relevant_info)
    
    prompt = f"""
    Query: {query}
    Relevant Information: {context}
    
    Please provide a response to the query based on the given information.
    """
    
    return gpt4_call(prompt)

def rag_over_tabular_with_gpt4(query: str, df: pd.DataFrame) -> str:
    """Implements RAG over tabular data using GPT-4."""
    context = f"Table data: {df.to_dict(orient='records')}"
    
    prompt = f"""
    Query: {query}
    {context}
    
    Please provide a response to the query based on the given tabular data.
    Perform any necessary calculations or analysis on the data to answer the query.
    """
    
    return gpt4_call(prompt)

def tag_with_gpt4(query: str, df: pd.DataFrame, knowledge_base: Dict[str, str]) -> str:
    """Implements TAG using GPT-4 for reasoning and response generation."""
    sql_query = text_to_sql(query)
    result_df = execute_sql(sql_query, df)
    
    sql_result = result_df.to_dict(orient='records')
    kb_info = [value for key, value in knowledge_base.items() if key in str(sql_result)]
    
    context = f"""
    SQL Query: {sql_query}
    Query Result: {sql_result}
    Additional Information: {' '.join(kb_info)}
    """
    
    prompt = f"""
    User Query: {query}
    Context:
    {context}
    
    Please provide a comprehensive answer to the user's query. Incorporate the SQL query results,
    additional information from the knowledge base, and any relevant reasoning or analysis.
    """
    
    return gpt4_call(prompt, "You are a helpful assistant with expertise in consumer electronics.")

# Demo
query = "What is the top selling laptop and why might it be popular?"

print("Text-to-SQL:")
print(text_to_sql(query))

print("\nRAG:")
print(rag_with_gpt4(query, knowledge_base))

print("\nRAG over Tabular Data:")
print(rag_over_tabular_with_gpt4(query, products_df))

print("\nTAG:")
print(tag_with_gpt4(query, products_df, knowledge_base))
