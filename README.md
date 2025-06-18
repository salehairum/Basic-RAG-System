# Basic RAG(Retrieval Augmented Generation) System
This is a simple task to understand how LLMs can be used in deployments, and how RAG systems are implemented.
This task utilizes Hugging Face library "transformers" and the model "google/flan-t5-base", with 4 sentences as contextual documents.

### OAuth
go to following url:

https://accounts.google.com/o/oauth2/v2/auth?client_id=client_id&redirect_uri=http://localhost:8000/oauth/callback&response_type=code&scope=openid%20email%20profile&access_type=offline

and copy token from there to send as a authorization token in future requests!
