import os
import sys
import io
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from sentence_transformers import CrossEncoder
from transformers import logging as transformers_logging

# ─────────────────────────────────────────────────────────────────────────────
# User interface Test
#
# This test does not involve AI.  It simply tests the User Interface in app.py
# ─────────────────────────────────────────────────────────────────────────────
print("\n")
print("Welcome to UI Test!\n")
print("-- Type 'end' to exit the assistant. --")

while True:
    user_input = input("\nEnter some text: ")
    if user_input.lower() == "end":
        print("Goodbye!")
        break
    
    print("Name:", user_input)