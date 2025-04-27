import { NextResponse } from 'next/server';

// Define the same structure as the Flask backend response
interface Prediction {
  patient_id: string;
  risk_assessment: {
    risk_level: string;
    risk_score: number;
    interpretation: string;
  };
  vital_statistics: {
    Patient_Demographics: {
      Age: string;
      Sex: string;
      Height: number;
      Weight: number;
      Pregnancy_Status: string;
    };
    Cardiovascular_Metrics: {
      Murmur_Assessment: {
        Presence: boolean;
        Location: string;
        Most_Audible_Location: string;
        Diastolic_Details: {
          Grade: string;
          Pitch: string;
          Quality: string;
          Shape: string;
          Timing: string;
        };
        Systolic_Details: {
          Grade: string;
          Pitch: string;
          Quality: string;
          Shape: string;
          Timing: string;
        };
      };
    };
  };
  detailed_analysis: {
    risk_factors: string[];
    cardiovascular_health: {
      murmur_status: string;
      murmur_location: string;
      overall_assessment: string;
    };
    lifestyle_recommendations: string[];
    warning_signs: string[];
    positive_indicators: string[];
    follow_up_recommendations: string[];
    diagnostic_tests: string[];
  };
  analysis_timestamp: string;
}

export async function POST(request: Request) {
  try {
    // Get patient_id from the request body
    const data = await request.json();
    const patientId = data.patient_id;

    if (!patientId) {
      return NextResponse.json(
        { error: "Missing 'patient_id' in request" },
        { status: 400 }
      );
    }

    // Forward the request to the Flask backend
    // Assuming Flask is running on localhost:10000
    const flaskResponse = await fetch('http://localhost:10000/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ patient_id: patientId }),
    });

    if (!flaskResponse.ok) {
      const errorData = await flaskResponse.json();
      return NextResponse.json(
        { error: errorData.error || 'Error processing request' },
        { status: flaskResponse.status }
      );
    }

    const result: Prediction = await flaskResponse.json();
    
    return NextResponse.json(result);
  } catch (error) {
    console.error('API route error:', error);
    return NextResponse.json(
      { error: 'Error processing request' },
      { status: 500 }
    );
  }
}

export async function GET(request: Request) {
  // Get patient_id from the URL params
  const { searchParams } = new URL(request.url);
  const patientId = searchParams.get('patient_id');

  if (!patientId) {
    return NextResponse.json(
      { error: "Missing 'patient_id' in request" },
      { status: 400 }
    );
  }

  try {
    // Forward the request to the Flask backend
    const flaskResponse = await fetch(`http://localhost:10000/predict?patient_id=${patientId}`, {
      method: 'GET',
    });

    if (!flaskResponse.ok) {
      const errorData = await flaskResponse.json();
      return NextResponse.json(
        { error: errorData.error || 'Error processing request' },
        { status: flaskResponse.status }
      );
    }

    const result: Prediction = await flaskResponse.json();
    
    return NextResponse.json(result);
  } catch (error) {
    console.error('API route error:', error);
    return NextResponse.json(
      { error: 'Error processing request' },
      { status: 500 }
    );
  }
} 