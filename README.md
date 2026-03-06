# 🎙️ Elite Voice POS: Intelligence-Driven Dining

"Turning Every Order into a High-Margin Transaction."

Elite Voice POS is a next-generation, AI-powered Point of Sale system designed to eliminate manual data entry and maximize restaurant profitability. By bridging real-time Voice Recognition with ML-driven Business Intelligence, the system identifies "Hidden Stars" (high-margin items) and suggests them to customers during the ordering process.

🚀 The Core Innovation: Two-Module Architecture

The system is split into two specialized engines that communicate via a secure real-time sync.

### 🧠 Module 1: The ML Revenue Engine

While traditional analytics focus on volume, our engine focuses on Profitability Velocity.

Contribution Margin Analysis: Calculates the net profit of every SKU.

Hidden Star Detection: Identifies items with high margins but low relative visibility.

Predictive Affinity: Cross-references historical sales to find high-probability pairings (e.g., Butter Chicken → Garlic Naan).

### 🗣️ Module 2: Voice Ordering Copilot

A high-performance interface that turns natural language into structured data.

Whisper STT Integration: Converts noisy restaurant audio into text.

Semantic Intent Mapping: Uses a Vector Database (ChromaDB) and Sentence Transformers to map fuzzy spoken words to exact Menu IDs.

Real-time Intelligence Sync: Receives "Push Notifications" from Module 1 to suggest upsells instantly on the server's dashboard.

### 🛠️ Tech Stack

Backend (Python/FastAPI)

FastAPI: High-performance asynchronous API framework.

ChromaDB: Vector database for semantic search and menu mapping.

Sentence-Transformers: all-MiniLM-L6-v2 for generating text embeddings.

OpenAI Whisper: State-of-the-art Speech-to-Text.

Frontend (React/TypeScript)

Vite: Lightning-fast build tool.

Tailwind CSS: Modern, responsive UI/UX.

Lucide React: Premium iconography.

### 📈 The Business Case: "Revenue Alpha"

The primary goal of Elite Voice POS is to generate Revenue Alpha—extra profit that would have been lost in a traditional manual POS.

Contextual Upselling: The system doesn't just ask "would you like a side?"; it asks for the specific side that Module 1 has identified as most profitable.

Speed of Service: Reduces order entry time by 40%, allowing for faster table turnover.

Data-Driven Menu: Automatically promotes items with high contribution margins.

## ## 🚦 Getting Started

Prerequisites

Python 3.9+

Node.js 18+

Installation

Clone the repository

git clone [https://github.com/aryandhawan/voice-pos.git](https://github.com/aryandhawan/voice-pos.git)
cd voice-pos


Setup Backend

cd core/backend-pos


python -m venv venv


source venv/bin/activate  # Or .\venv\Scripts\activate on Windows


pip install -r requirements.txt


uvicorn main:app --reload


Setup Frontend

cd frontend
npm install
npm run dev


🛡️ Security & Scalability

Webhook Security: Communication between modules is secured via X-API-Key validation.

Persistence: ChromaDB ensures that ML intelligence persists even after system restarts.

Edge-Ready: Designed to run locally in-restaurant to ensure zero-latency voice processing.

Developed for Hackathon 2026 Built with ❤️ by Aryan Dhawan
