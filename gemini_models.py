import google.generativeai as genai
genai.configure(api_key="AQ.Ab8RN6LDsa8QN7gjzFW3LsB8QHkTf8uytZBYZL7_AHcCef4e5g")
for m in genai.list_models():
    print(m.name)