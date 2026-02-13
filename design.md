# DharmaJyothi - System Design Document

## 1. Architecture Overview

### 1.1 System Architecture Pattern
DharmaJyothi follows a **microservices architecture** with clear separation of concerns, enabling scalability, maintainability, and independent deployment of different system components.

### 1.2 High-Level Architecture Diagram
```
┌─────────────────────────────────────────────────────────────────┐
│                    Client Applications                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │   Web App   │  │ Mobile App  │  │    Admin Dashboard      │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────────────┐
                    │   API Gateway   │
                    │   (Node.js)     │
                    └─────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                    Microservices Layer                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │    User     │  │  Document   │  │      AI Engine         │ │
│  │  Service    │  │  Service    │  │     Service             │ │
│  │ (Node.js)   │  │ (Node.js)   │  │    (Node.js)            │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ Geolocation │  │ Notification│  │    Payment              │ │
│  │  Service    │  │  Service    │  │    Service              │ │
│  │ (Node.js)   │  │ (Node.js)   │  │   (Node.js)             │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                    Data & Storage Layer                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ PostgreSQL  │  │   Redis     │  │    File Storage         │ │
│  │  Database   │  │   Cache     │  │    (Cloud Storage)      │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 2. Technology Stack Mapping

### 2.1 Infrastructure & Backend Technologies

#### 2.1.1 Runtime Environment
- **Technology**: Node.js
- **Purpose**: Primary backend runtime for all microservices
- **Implementation**: 
  - Express.js framework for REST API development
  - TypeScript for type-safe development
  - PM2 for process management and clustering
  - Docker containers for deployment consistency

#### 2.1.2 Database System
- **Technology**: PostgreSQL
- **Purpose**: Primary relational database for structured data
- **Schema Design**:
  ```sql
  -- Core Tables
  Users (user_id, email, password_hash, user_type, created_at)
  Lawyers (lawyer_id, user_id, specialization, credentials, location)
  Cases (case_id, client_id, lawyer_id, status, created_at)
  Documents (doc_id, case_id, file_path, analysis_result, upload_date)
  Chat_Sessions (session_id, user_id, conversation_data, created_at)
  Geolocation_Data (location_id, lawyer_id, latitude, longitude, address)
  ```

#### 2.1.3 Document Processing Technologies
- **Tesseract OCR**
  - **Module**: Document Processing Service
  - **Purpose**: Text extraction from scanned images and PDFs
  - **Implementation**: Node.js wrapper (node-tesseract-ocr)
  - **Use Cases**: Processing uploaded legal documents, court papers, contracts

- **Google Cloud Vision API**
  - **Module**: Enhanced OCR Service
  - **Purpose**: High-accuracy text extraction and document analysis
  - **Implementation**: REST API integration with authentication
  - **Use Cases**: Complex document layouts, handwritten text recognition

- **PDF.js**
  - **Module**: Document Viewer Service
  - **Purpose**: Client-side PDF rendering and text extraction
  - **Implementation**: Browser-based PDF processing
  - **Use Cases**: Real-time document preview, text selection, annotation

- **Apache POI**
  - **Module**: Document Parser Service (Java microservice)
  - **Purpose**: Advanced document format processing
  - **Implementation**: Java-based service with REST API
  - **Use Cases**: Complex Word documents, Excel spreadsheets, PowerPoint presentations

### 2.2 Intelligence & AI Engine Technologies

#### 2.2.1 Large Language Models
- **GPT-5**
  - **Module**: Primary AI Legal Assistant
  - **Purpose**: Advanced legal reasoning and document analysis
  - **Implementation**: OpenAI API integration with custom prompts
  - **Use Cases**: Complex legal queries, contract analysis, legal strategy

- **Llama**
  - **Module**: Secondary AI Engine
  - **Purpose**: Backup reasoning engine and specialized legal tasks
  - **Implementation**: Self-hosted model with Hugging Face Transformers
  - **Use Cases**: Privacy-sensitive queries, offline processing, cost optimization

#### 2.2.2 Machine Learning & NLP Technologies
- **Scikit-learn**
  - **Module**: Document Classification Service
  - **Purpose**: Legal document type classification and categorization
  - **Implementation**: Python microservice with REST API
  - **Models**: Random Forest, SVM for document classification
  - **Use Cases**: Automatic document tagging, case type identification

- **TensorFlow**
  - **Module**: Deep Learning Service
  - **Purpose**: Advanced NLP tasks and custom model training
  - **Implementation**: TensorFlow Serving with Python backend
  - **Models**: Custom neural networks for legal entity recognition
  - **Use Cases**: Legal entity extraction, sentiment analysis, risk assessment

- **Hugging Face Transformers**
  - **Module**: Semantic Search Service
  - **Purpose**: Document embeddings and similarity search
  - **Implementation**: Python service with transformer models
  - **Models**: BERT, RoBERTa for legal text understanding
  - **Use Cases**: Similar case finding, legal precedent matching, semantic search

### 2.3 Geospatial Technologies

#### 2.3.1 Mapping and Location Services
- **Google Maps API**
  - **Module**: Primary Geolocation Service
  - **Purpose**: Lawyer location mapping and route optimization
  - **Implementation**: JavaScript SDK and REST API integration
  - **Features**: Geocoding, reverse geocoding, distance calculation, map visualization

- **OpenStreetMap (OSM)**
  - **Module**: Alternative Mapping Service
  - **Purpose**: Backup mapping solution and cost optimization
  - **Implementation**: Leaflet.js with OSM tiles
  - **Features**: Open-source mapping, custom tile servers, offline capability

## 3. Detailed System Design

### 3.1 User Management Service

#### 3.1.1 Architecture
```javascript
// Node.js Express Service Structure
const express = require('express');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const { Pool } = require('pg');

class UserService {
  constructor() {
    this.db = new Pool({
      connectionString: process.env.DATABASE_URL
    });
  }
  
  async registerUser(userData) {
    // User registration logic with PostgreSQL
  }
  
  async authenticateUser(credentials) {
    // JWT-based authentication
  }
}
```

#### 3.1.2 Database Schema
```sql
-- PostgreSQL Schema for User Management
CREATE TABLE users (
  user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email VARCHAR(255) UNIQUE NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  user_type ENUM('client', 'lawyer', 'admin') NOT NULL,
  profile_data JSONB,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE lawyers (
  lawyer_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(user_id),
  bar_number VARCHAR(50) UNIQUE,
  specializations TEXT[],
  credentials JSONB,
  verification_status VARCHAR(20) DEFAULT 'pending',
  practice_areas TEXT[]
);
```

### 3.2 Document Processing Service

#### 3.2.1 OCR Processing Pipeline
```javascript
// Document Processing with Tesseract and Google Vision
const tesseract = require('node-tesseract-ocr');
const vision = require('@google-cloud/vision');

class DocumentProcessor {
  constructor() {
    this.visionClient = new vision.ImageAnnotatorClient();
    this.tesseractConfig = {
      lang: 'eng',
      oem: 1,
      psm: 3,
    };
  }
  
  async processDocument(filePath, useCloudVision = false) {
    if (useCloudVision) {
      return await this.processWithGoogleVision(filePath);
    } else {
      return await this.processWithTesseract(filePath);
    }
  }
  
  async processWithTesseract(filePath) {
    try {
      const text = await tesseract.recognize(filePath, this.tesseractConfig);
      return {
        extractedText: text,
        confidence: 'medium',
        processor: 'tesseract'
      };
    } catch (error) {
      throw new Error(`Tesseract processing failed: ${error.message}`);
    }
  }
  
  async processWithGoogleVision(filePath) {
    const [result] = await this.visionClient.textDetection(filePath);
    const detections = result.textAnnotations;
    return {
      extractedText: detections[0]?.description || '',
      confidence: 'high',
      processor: 'google-vision'
    };
  }
}
```

#### 3.2.2 PDF Processing Integration
```javascript
// PDF.js Integration for Client-Side Processing
const pdfjsLib = require('pdfjs-dist/legacy/build/pdf');

class PDFProcessor {
  async extractTextFromPDF(pdfBuffer) {
    const pdf = await pdfjsLib.getDocument({data: pdfBuffer}).promise;
    let fullText = '';
    
    for (let i = 1; i <= pdf.numPages; i++) {
      const page = await pdf.getPage(i);
      const textContent = await page.getTextContent();
      const pageText = textContent.items.map(item => item.str).join(' ');
      fullText += pageText + '\n';
    }
    
    return fullText;
  }
}
```

### 3.3 AI Legal Assistant Service

#### 3.3.1 LLM Integration Architecture
```javascript
// GPT-5 and Llama Integration
const OpenAI = require('openai');
const { HfInference } = require('@huggingface/inference');

class AILegalAssistant {
  constructor() {
    this.openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY
    });
    this.hf = new HfInference(process.env.HF_API_KEY);
  }
  
  async processLegalQuery(query, context, model = 'gpt5') {
    const systemPrompt = `You are a legal AI assistant. Provide accurate legal information based on the context provided. Always include disclaimers about seeking professional legal advice.`;
    
    if (model === 'gpt5') {
      return await this.queryGPT5(systemPrompt, query, context);
    } else {
      return await this.queryLlama(systemPrompt, query, context);
    }
  }
  
  async queryGPT5(systemPrompt, query, context) {
    const response = await this.openai.chat.completions.create({
      model: "gpt-5",
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: `Context: ${context}\n\nQuery: ${query}` }
      ],
      temperature: 0.3,
      max_tokens: 1000
    });
    
    return response.choices[0].message.content;
  }
  
  async queryLlama(systemPrompt, query, context) {
    const prompt = `${systemPrompt}\n\nContext: ${context}\n\nQuery: ${query}\n\nResponse:`;
    
    const response = await this.hf.textGeneration({
      model: 'meta-llama/Llama-2-70b-chat-hf',
      inputs: prompt,
      parameters: {
        max_new_tokens: 1000,
        temperature: 0.3,
        return_full_text: false
      }
    });
    
    return response.generated_text;
  }
}
```

#### 3.3.2 ML Classification Service
```python
# Scikit-learn Document Classification
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

class LegalDocumentClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000, stop_words='english')),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        self.categories = [
            'contract', 'legal_notice', 'court_document', 
            'agreement', 'will', 'power_of_attorney'
        ]
    
    def train_model(self, training_data):
        X = training_data['text']
        y = training_data['category']
        self.pipeline.fit(X, y)
        joblib.dump(self.pipeline, 'legal_classifier_model.pkl')
    
    def classify_document(self, text):
        if not hasattr(self, 'pipeline'):
            self.pipeline = joblib.load('legal_classifier_model.pkl')
        
        prediction = self.pipeline.predict([text])[0]
        probabilities = self.pipeline.predict_proba([text])[0]
        
        return {
            'category': prediction,
            'confidence': max(probabilities),
            'all_probabilities': dict(zip(self.categories, probabilities))
        }
```

#### 3.3.3 TensorFlow Deep Learning Service
```python
# TensorFlow Legal Entity Recognition
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel
import numpy as np

class LegalEntityRecognizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')
        self.model = TFAutoModel.from_pretrained('nlpaueb/legal-bert-base-uncased')
        
    def extract_legal_entities(self, text):
        # Tokenize input text
        inputs = self.tokenizer(text, return_tensors='tf', truncation=True, padding=True)
        
        # Get model outputs
        outputs = self.model(**inputs)
        
        # Process embeddings for entity recognition
        embeddings = outputs.last_hidden_state
        
        # Custom entity recognition logic
        entities = self._identify_entities(embeddings, text)
        
        return entities
    
    def _identify_entities(self, embeddings, text):
        # Custom logic for legal entity identification
        # This would include person names, organizations, dates, amounts, etc.
        entities = {
            'persons': [],
            'organizations': [],
            'dates': [],
            'monetary_amounts': [],
            'legal_references': []
        }
        
        return entities
```

### 3.4 Geolocation Service

#### 3.4.1 Google Maps Integration
```javascript
// Google Maps API Integration
const { Client } = require('@googlemaps/google-maps-services-js');

class GeolocationService {
  constructor() {
    this.mapsClient = new Client({});
    this.apiKey = process.env.GOOGLE_MAPS_API_KEY;
  }
  
  async findNearbyLawyers(userLocation, specialization, radius = 10000) {
    try {
      // Geocode user location
      const geocodeResponse = await this.mapsClient.geocode({
        params: {
          address: userLocation,
          key: this.apiKey
        }
      });
      
      const userCoords = geocodeResponse.data.results[0].geometry.location;
      
      // Query database for lawyers within radius
      const nearbyLawyers = await this.queryLawyersInRadius(
        userCoords.lat, 
        userCoords.lng, 
        radius, 
        specialization
      );
      
      // Calculate distances and add route information
      const lawyersWithDistance = await Promise.all(
        nearbyLawyers.map(async (lawyer) => {
          const distance = await this.calculateDistance(userCoords, lawyer.location);
          return {
            ...lawyer,
            distance: distance,
            estimatedTravelTime: await this.getEstimatedTravelTime(userCoords, lawyer.location)
          };
        })
      );
      
      return lawyersWithDistance.sort((a, b) => a.distance - b.distance);
      
    } catch (error) {
      throw new Error(`Geolocation service error: ${error.message}`);
    }
  }
  
  async queryLawyersInRadius(lat, lng, radius, specialization) {
    const query = `
      SELECT l.*, u.email, u.profile_data,
             ST_Distance(
               ST_GeogFromText('POINT(' || gl.longitude || ' ' || gl.latitude || ')'),
               ST_GeogFromText('POINT(' || $2 || ' ' || $1 || ')')
             ) as distance
      FROM lawyers l
      JOIN users u ON l.user_id = u.user_id
      JOIN geolocation_data gl ON l.lawyer_id = gl.lawyer_id
      WHERE ST_DWithin(
        ST_GeogFromText('POINT(' || gl.longitude || ' ' || gl.latitude || ')'),
        ST_GeogFromText('POINT(' || $2 || ' ' || $1 || ')'),
        $3
      )
      AND ($4 IS NULL OR $4 = ANY(l.specializations))
      ORDER BY distance;
    `;
    
    const result = await this.db.query(query, [lat, lng, radius, specialization]);
    return result.rows;
  }
  
  async calculateDistance(origin, destination) {
    const response = await this.mapsClient.distancematrix({
      params: {
        origins: [`${origin.lat},${origin.lng}`],
        destinations: [`${destination.lat},${destination.lng}`],
        units: 'metric',
        key: this.apiKey
      }
    });
    
    return response.data.rows[0].elements[0].distance.value;
  }
}
```

#### 3.4.2 OpenStreetMap Integration
```javascript
// OpenStreetMap Alternative Implementation
const axios = require('axios');

class OSMGeolocationService {
  constructor() {
    this.nominatimBaseUrl = 'https://nominatim.openstreetmap.org';
    this.overpassBaseUrl = 'https://overpass-api.de/api/interpreter';
  }
  
  async geocodeAddress(address) {
    try {
      const response = await axios.get(`${this.nominatimBaseUrl}/search`, {
        params: {
          q: address,
          format: 'json',
          limit: 1
        }
      });
      
      if (response.data.length > 0) {
        return {
          lat: parseFloat(response.data[0].lat),
          lng: parseFloat(response.data[0].lon)
        };
      }
      
      throw new Error('Address not found');
    } catch (error) {
      throw new Error(`Geocoding failed: ${error.message}`);
    }
  }
  
  async findNearbyPlaces(lat, lng, radius, amenity = 'office') {
    const overpassQuery = `
      [out:json][timeout:25];
      (
        node["amenity"="${amenity}"](around:${radius},${lat},${lng});
        way["amenity"="${amenity}"](around:${radius},${lat},${lng});
        relation["amenity"="${amenity}"](around:${radius},${lat},${lng});
      );
      out center;
    `;
    
    try {
      const response = await axios.post(this.overpassBaseUrl, overpassQuery, {
        headers: { 'Content-Type': 'text/plain' }
      });
      
      return response.data.elements;
    } catch (error) {
      throw new Error(`Overpass query failed: ${error.message}`);
    }
  }
}
```

## 4. Data Architecture

### 4.1 PostgreSQL Database Design

#### 4.1.1 Core Schema
```sql
-- Complete PostgreSQL Schema
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "postgis";

-- Users and Authentication
CREATE TABLE users (
  user_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  email VARCHAR(255) UNIQUE NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  user_type VARCHAR(20) CHECK (user_type IN ('client', 'lawyer', 'admin')),
  profile_data JSONB DEFAULT '{}',
  is_verified BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Lawyer-specific information
CREATE TABLE lawyers (
  lawyer_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID REFERENCES users(user_id) ON DELETE CASCADE,
  bar_number VARCHAR(50) UNIQUE,
  specializations TEXT[] DEFAULT '{}',
  credentials JSONB DEFAULT '{}',
  verification_status VARCHAR(20) DEFAULT 'pending',
  practice_areas TEXT[] DEFAULT '{}',
  hourly_rate DECIMAL(10,2),
  availability_schedule JSONB DEFAULT '{}',
  rating DECIMAL(3,2) DEFAULT 0.00,
  total_reviews INTEGER DEFAULT 0
);

-- Geolocation data
CREATE TABLE geolocation_data (
  location_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  lawyer_id UUID REFERENCES lawyers(lawyer_id) ON DELETE CASCADE,
  address TEXT NOT NULL,
  city VARCHAR(100),
  state VARCHAR(50),
  country VARCHAR(50),
  postal_code VARCHAR(20),
  coordinates GEOGRAPHY(POINT, 4326),
  service_radius INTEGER DEFAULT 50000 -- in meters
);

-- Legal cases
CREATE TABLE cases (
  case_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  client_id UUID REFERENCES users(user_id),
  lawyer_id UUID REFERENCES lawyers(lawyer_id),
  case_type VARCHAR(100),
  status VARCHAR(50) DEFAULT 'open',
  description TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  closed_at TIMESTAMP WITH TIME ZONE
);

-- Document storage and analysis
CREATE TABLE documents (
  document_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  case_id UUID REFERENCES cases(case_id),
  uploader_id UUID REFERENCES users(user_id),
  file_name VARCHAR(255) NOT NULL,
  file_path TEXT NOT NULL,
  file_size BIGINT,
  mime_type VARCHAR(100),
  ocr_text TEXT,
  analysis_result JSONB DEFAULT '{}',
  document_type VARCHAR(100),
  upload_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  processed_at TIMESTAMP WITH TIME ZONE
);

-- AI Chat sessions
CREATE TABLE chat_sessions (
  session_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID REFERENCES users(user_id),
  case_id UUID REFERENCES cases(case_id),
  conversation_data JSONB DEFAULT '[]',
  session_metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  last_activity TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_lawyers_specializations ON lawyers USING GIN(specializations);
CREATE INDEX idx_geolocation_coordinates ON geolocation_data USING GIST(coordinates);
CREATE INDEX idx_documents_case_id ON documents(case_id);
CREATE INDEX idx_chat_sessions_user_id ON chat_sessions(user_id);
CREATE INDEX idx_cases_status ON cases(status);
```

### 4.2 Caching Strategy with Redis
```javascript
// Redis Caching Implementation
const redis = require('redis');

class CacheService {
  constructor() {
    this.client = redis.createClient({
      host: process.env.REDIS_HOST,
      port: process.env.REDIS_PORT,
      password: process.env.REDIS_PASSWORD
    });
  }
  
  async cacheDocumentAnalysis(documentId, analysis) {
    const key = `doc_analysis:${documentId}`;
    await this.client.setex(key, 3600, JSON.stringify(analysis)); // 1 hour TTL
  }
  
  async getCachedAnalysis(documentId) {
    const key = `doc_analysis:${documentId}`;
    const cached = await this.client.get(key);
    return cached ? JSON.parse(cached) : null;
  }
  
  async cacheLawyerSearch(searchParams, results) {
    const key = `lawyer_search:${JSON.stringify(searchParams)}`;
    await this.client.setex(key, 300, JSON.stringify(results)); // 5 minutes TTL
  }
}
```

## 5. Security Architecture

### 5.1 Authentication & Authorization
```javascript
// JWT-based Authentication
const jwt = require('jsonwebtoken');
const bcrypt = require('bcrypt');

class AuthService {
  constructor() {
    this.jwtSecret = process.env.JWT_SECRET;
    this.jwtExpiry = process.env.JWT_EXPIRY || '24h';
  }
  
  async hashPassword(password) {
    return await bcrypt.hash(password, 12);
  }
  
  async verifyPassword(password, hash) {
    return await bcrypt.compare(password, hash);
  }
  
  generateToken(user) {
    return jwt.sign(
      { 
        userId: user.user_id, 
        userType: user.user_type,
        email: user.email 
      },
      this.jwtSecret,
      { expiresIn: this.jwtExpiry }
    );
  }
  
  verifyToken(token) {
    return jwt.verify(token, this.jwtSecret);
  }
}

// Role-based Access Control Middleware
const rbacMiddleware = (allowedRoles) => {
  return (req, res, next) => {
    const userType = req.user.userType;
    
    if (!allowedRoles.includes(userType)) {
      return res.status(403).json({ error: 'Access denied' });
    }
    
    next();
  };
};
```

### 5.2 Data Encryption
```javascript
// Document Encryption Service
const crypto = require('crypto');

class EncryptionService {
  constructor() {
    this.algorithm = 'aes-256-gcm';
    this.secretKey = process.env.ENCRYPTION_KEY;
  }
  
  encrypt(text) {
    const iv = crypto.randomBytes(16);
    const cipher = crypto.createCipher(this.algorithm, this.secretKey);
    cipher.setAAD(Buffer.from('DharmaJyothi', 'utf8'));
    
    let encrypted = cipher.update(text, 'utf8', 'hex');
    encrypted += cipher.final('hex');
    
    const authTag = cipher.getAuthTag();
    
    return {
      encrypted,
      iv: iv.toString('hex'),
      authTag: authTag.toString('hex')
    };
  }
  
  decrypt(encryptedData) {
    const decipher = crypto.createDecipher(this.algorithm, this.secretKey);
    decipher.setAAD(Buffer.from('DharmaJyothi', 'utf8'));
    decipher.setAuthTag(Buffer.from(encryptedData.authTag, 'hex'));
    
    let decrypted = decipher.update(encryptedData.encrypted, 'hex', 'utf8');
    decrypted += decipher.final('utf8');
    
    return decrypted;
  }
}
```

## 6. Deployment Architecture

### 6.1 Docker Configuration
```dockerfile
# Node.js Service Dockerfile
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --only=production

COPY . .

EXPOSE 3000

CMD ["node", "server.js"]
```

### 6.2 Docker Compose Setup
```yaml
# docker-compose.yml
version: '3.8'

services:
  api-gateway:
    build: ./api-gateway
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
    depends_on:
      - postgres
      - redis

  user-service:
    build: ./user-service
    environment:
      - DATABASE_URL=${DATABASE_URL}
    depends_on:
      - postgres

  document-service:
    build: ./document-service
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - GOOGLE_CLOUD_KEY=${GOOGLE_CLOUD_KEY}
    depends_on:
      - postgres

  ai-service:
    build: ./ai-service
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - HF_API_KEY=${HF_API_KEY}

  geolocation-service:
    build: ./geolocation-service
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - GOOGLE_MAPS_API_KEY=${GOOGLE_MAPS_API_KEY}
    depends_on:
      - postgres

  postgres:
    image: postgis/postgis:14-3.2
    environment:
      - POSTGRES_DB=dharmajyothi
      - POSTGRES_USER=${DB_USER}
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

## 7. Performance Optimization

### 7.1 Database Optimization
- **Connection Pooling**: PostgreSQL connection pools for each service
- **Query Optimization**: Proper indexing and query analysis
- **Read Replicas**: Separate read replicas for analytics and reporting
- **Partitioning**: Table partitioning for large datasets (documents, chat logs)

### 7.2 Caching Strategy
- **Redis Caching**: Frequently accessed data and search results
- **CDN Integration**: Static assets and document thumbnails
- **Application-level Caching**: In-memory caching for configuration data

### 7.3 API Optimization
- **Rate Limiting**: Prevent API abuse and ensure fair usage
- **Response Compression**: Gzip compression for API responses
- **Pagination**: Efficient pagination for large result sets
- **Async Processing**: Background job processing for heavy tasks

## 8. Monitoring & Observability

### 8.1 Logging Strategy
```javascript
// Structured Logging with Winston
const winston = require('winston');

const logger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  defaultMeta: { service: 'dharmajyothi-api' },
  transports: [
    new winston.transports.File({ filename: 'error.log', level: 'error' }),
    new winston.transports.File({ filename: 'combined.log' }),
    new winston.transports.Console({
      format: winston.format.simple()
    })
  ]
});
```

### 8.2 Health Checks
```javascript
// Health Check Endpoints
app.get('/health', async (req, res) => {
  const health = {
    status: 'OK',
    timestamp: new Date().toISOString(),
    services: {
      database: await checkDatabaseHealth(),
      redis: await checkRedisHealth(),
      external_apis: await checkExternalAPIs()
    }
  };
  
  res.json(health);
});
```

This comprehensive design document maps all the specified technologies to their respective modules and provides detailed implementation guidance for the DharmaJyothi legal-tech platform.