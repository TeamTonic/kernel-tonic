#!/usr/bin/env python3
"""
Test script for vLLM inference with Kernel Tonic model.
"""

import requests
import json
import time

def test_vllm_inference():
    """Test vLLM inference API."""
    url = "http://localhost:8000/v1/completions"
    
    # Test prompt
    prompt = "The future of artificial intelligence is"
    
    payload = {
        "model": "kernel-tonic",
        "prompt": prompt,
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9,
        "stream": False
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        print("Testing vLLM inference...")
        print(f"Prompt: {prompt}")
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            completion = result['choices'][0]['text']
            print(f"Generated text: {completion}")
            print("✅ Inference test successful!")
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to vLLM server")
        print("Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_chat_completion():
    """Test chat completion API."""
    url = "http://localhost:8000/v1/chat/completions"
    
    payload = {
        "model": "kernel-tonic",
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        print("\nTesting chat completion...")
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            message = result['choices'][0]['message']['content']
            print(f"Assistant: {message}")
            print("✅ Chat completion test successful!")
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to vLLM server")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Run all tests."""
    print("=== Kernel Tonic vLLM Inference Test ===")
    
    # Wait a bit for server to start
    print("Waiting for server to be ready...")
    time.sleep(5)
    
    # Test completion
    test_vllm_inference()
    
    # Test chat completion
    test_chat_completion()
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main() 