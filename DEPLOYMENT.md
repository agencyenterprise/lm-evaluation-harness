# Deploying to Railway

This guide explains how to deploy the Moral Stories Evaluation API to Railway.

## Prerequisites

1. A [Railway](https://railway.app/) account
2. [Railway CLI](https://docs.railway.app/develop/cli) installed (optional, but recommended)
3. OpenAI API key
4. MongoDB URI (optional, for database integration)

## Deployment Steps

### Option 1: Deploy via Railway Dashboard

1. **Create a new project on Railway**
   - Go to [Railway Dashboard](https://railway.app/dashboard)
   - Click "New Project" > "Deploy from GitHub repo"
   - Connect your GitHub account and select your repository

2. **Set Environment Variables**
   - In your project settings, go to "Variables"
   - Add the following variables:
     ```
     OPENAI_API_KEY=your_openai_api_key
     MONGODB_URI=your_mongodb_connection_string (optional)
     ```

3. **Deploy**
   - Railway will automatically detect your Procfile and start the deployment
   - Once deployed, you'll get a public URL for your API

### Option 2: Deploy via Railway CLI

1. **Login to Railway**
   ```bash
   railway login
   ```

2. **Link to your project**
   ```bash
   # Create a new project
   railway init
   
   # Or link to existing project
   railway link
   ```

3. **Set environment variables**
   ```bash
   railway variables set OPENAI_API_KEY=your_openai_api_key
   railway variables set MONGODB_URI=your_mongodb_connection_string
   ```

4. **Deploy your project**
   ```bash
   railway up
   ```

5. **Open your deployed API**
   ```bash
   railway open
   ```

## Using Your Deployed API

Once deployed, your API will be available at the URL provided by Railway. You can interact with it using the following endpoints:

### 1. Check if API is running
```
GET /
```

### 2. Start an evaluation
```
POST /evaluate
```
Example request body:
```json
{
  "model": "gpt-4o",
  "examples": 5,
  "message_id": "message_123",
  "system": "You are an ethical assistant."
}
```

### 3. Get evaluation results
```
GET /result/{task_id}
```

## Connecting from Your Frontend Application

To use this API from your frontend application, make HTTP requests to the deployed API:

```javascript
// Start an evaluation
const response = await fetch('https://your-railway-url.railway.app/evaluate', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    model: 'gpt-4o',
    examples: 5,
    message_id: 'conversation_123',
    context: [
      { role: 'system', content: 'You are an ethical assistant.' },
      { role: 'user', content: 'Is it ethical to break a promise?' }
    ]
  })
});

const { task_id } = await response.json();

// Poll for results
const checkResult = async () => {
  const resultResponse = await fetch(`https://your-railway-url.railway.app/result/${task_id}`);
  const result = await resultResponse.json();
  
  if (result.status === 'processing') {
    // Try again in 5 seconds
    setTimeout(checkResult, 5000);
  } else {
    // Use the completed results
    console.log(result);
  }
};

checkResult();
```

## Troubleshooting

1. **API not starting**: Check the logs in the Railway dashboard to identify any issues.
2. **Missing dependencies**: Make sure all dependencies are in the requirements.txt file.
3. **Environment variables**: Verify that all required environment variables are set correctly.
4. **Memory issues**: If your evaluations require more memory, adjust the service size in Railway settings. 