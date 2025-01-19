import torch
from BD_memory_utils import store_message, retrieve_history
from utils import remove_special_characters
import ollama

def query_ollama_with_memory(user_message, model="llama3.2"):
    # Recupera o histórico da conversa
    history = retrieve_history()
    context = "\n".join([f"Usuário: {msg[0]}\nBot: {msg[1]}" for msg in history])

    # Convert history to message format
    history_messages = []
    for i, msg in enumerate(history):
        history_messages.append({
            'role': 'user' if i % 2 == 0 else 'assistant',
            'content': msg[0] if i % 2 == 0 else msg[1]
        })
    
    # Create messages list with system prompt and history
    system_prompt = f"""Você é um robô de compania em um hospital.
    Histórico da conversa:
    {context}"""
    
    messages = [
        {'role': 'system', 'content': system_prompt},
        *history_messages,
        {'role': 'user', 'content': user_message}
    ]
    
    # Make the call to Ollama with the conversation history
    response = ollama.chat(
        model=model,
        messages=messages
    )

    # Extraindo a resposta do Ollama
    bot_response = response['message']['content']
    
    # Armazena a mensagem atual no banco de dados
    store_message(user_message, bot_response)

    print('inferencia concluida')
    
    return remove_special_characters(bot_response)