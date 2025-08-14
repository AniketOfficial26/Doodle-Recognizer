import requests
import json

def test_kie_ai_api():
    url = "https://4o-api.kie.ai/v1/image"
    
    payload = {
        "prompt": "A colorful, fun cartoon illustration of a cat that would appeal to children. "
                 "cartoon style, bright and vibrant colors, cute and friendly appearance, "
                 "exaggerated features, simple shapes, children's storybook style, solid light background. "
                 "The cat should be the main focus, cheerful and engaging for kids.",
        "size": "512x512"
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        print("Sending request to Kie.AI 4o Image API...")
        response = requests.post(url, json=payload, headers=headers)
        
        print(f"Status Code: {response.status_code}")
        print("Response Headers:")
        for header, value in response.headers.items():
            print(f"  {header}: {value}")
            
        print("\nResponse Body:")
        try:
            print(json.dumps(response.json(), indent=2))
        except:
            print(response.text)
            
    except Exception as e:
        print(f"Error making request: {str(e)}")

if __name__ == "__main__":
    test_kie_ai_api()
