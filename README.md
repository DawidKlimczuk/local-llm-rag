# Local LLM & RAG Pipeline (TypeScript + Ollama)

W pełni lokalna, autonomiczna implementacja architektury Retrieval-Augmented Generation (RAG) działająca w środowisku on-premise bez konieczności wysyłania danych do chmurowych interfejsów API. Projekt umożliwia semantyczne odpytywanie wewnętrznej bazy wiedzy przy zachowaniu pełnej prywatności danych i eliminacji halucynacji modelu językowego.

---

## 🚀 Technologie i Narzędzia

* **Środowisko uruchomieniowe:** Node.js, TypeScript (tsx)
* **Lokalny serwer modeli:** Ollama
* **Model językowy (LLM):** Meta Llama 3.2 (3B)
* **Model wektorowy (Embeddings):** nomic-embed-text
* **Przetwarzanie tekstu:** @langchain/textsplitters
* **Wyszukiwanie semantyczne:** Implementacja podobieństwa cosinusowego (Cosine Similarity)

---

## 🛠️ Architektura i Zasada Działania

Projekt realizuje pełny cykl przetwarzania danych w procesie RAG:

1. **Ingestia danych i chunking:** Wczytanie nieustrukturyzowanego pliku tekstowego (`data/knowledge.txt`) oraz podział na nakładające się fragmenty za pomocą RecursiveCharacterTextSplitter.
2. **Wektoryzacja (Embeddings):** Przekształcenie fragmentów tekstu w wielowymiarowe wektory semantyczne przy użyciu lokalnego modelu nomic-embed-text.
3. **Wyszukiwanie kontekstu:** Wektoryzacja zapytania użytkownika i wyznaczenie najbardziej trafnych fragmentów tekstu metodą podobieństwa cosinusowego (Top-K).
4. **Augmentacja i generacja:** Przekazanie wyselekcjonowanego kontekstu wraz ze ścisłymi instrukcjami systemowymi do lokalnego modelu Llama 3.2 w celu wygenerowania faktograficznej odpowiedzi.

---

## 📸 Przykładowe Działanie

System prawidłowo identyfikuje procedury awaryjne oraz tokeny dostępu zawarte w dokumentacji, nie dopuszczając do zmyślania informacji:
<img width="1055" height="495" alt="image" src="https://github.com/user-attachments/assets/eed917a1-afb2-403d-a95d-9b71d476b144" />

---

## 📦 Wymagania i Uruchomienie
```bash
### 1. Wymagania wstępne
Zainstaluj narzędzie Ollama i pobierz wymagane modele lokalne:
ollama pull llama3.2
ollama pull nomic-embed-text

2. Instalacja zależności
Bash
git clone [https://github.com/DawidKlimczuk/local-llm-rag.git](https://github.com/DawidKlimczuk/local-llm-rag.git)
cd local-llm-rag
npm install

3. Uruchomienie pipeline'u
Bash
npm start
