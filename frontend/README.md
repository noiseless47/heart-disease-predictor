# Heart Disease Prediction App

This application provides a heart disease risk assessment based on patient data. It consists of a Flask backend for the machine learning model and a Next.js frontend for the user interface.

## Project Structure

- `frontend/`: Next.js frontend application
- Root directory: Flask backend application, machine learning model, and data processing

## Setup and Running

### Backend (Flask)

1. Install the required Python packages:

```bash
pip install -r requirements.txt
```

2. Run the Flask backend:

```bash
python app.py
```

The Flask backend will run on http://localhost:5001

### Frontend (Next.js)

1. Navigate to the frontend directory:

```bash
cd frontend
```

2. Install the required Node.js packages:

```bash
npm install
```

3. Run the Next.js frontend:

```bash
npm run dev
```

The Next.js frontend will run on http://localhost:3000

## How It Works

1. The Flask backend serves the machine learning model that predicts heart disease risk based on patient data.
2. The Next.js frontend provides a user interface to enter a patient ID and view the prediction results.
3. The frontend communicates with the backend via API calls to get predictions.

## Features

- Heart disease risk prediction
- Detailed patient information display
- Risk factor analysis
- Lifestyle recommendations
- Follow-up recommendations
- Diagnostic test suggestions

## API Endpoints

- `POST /predict`: Get prediction for a patient based on patient_id
- `GET /predict?patient_id=<ID>`: Get prediction for a patient using query parameters

This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
