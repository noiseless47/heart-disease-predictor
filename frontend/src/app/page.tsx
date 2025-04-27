'use client'

import { useState } from 'react'
import ThemeToggle from './components/ThemeToggle'
import { useTheme } from './ThemeProvider'

// Helper function to try parsing a JSON string as an array
const tryParseJsonArray = (jsonString: string): string[] => {
  try {
    const parsed = JSON.parse(jsonString);
    if (Array.isArray(parsed)) {
      return parsed;
    }
    return [jsonString];
  } catch (e) {
    // If it's not valid JSON, treat it as a single string
    return [jsonString];
  }
};

// Helper function to format object keys for display
const formatKeyName = (key: string): string => {
  return key
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
};

export default function Home() {
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<any>(null)
  const [patientId, setPatientId] = useState('')
  const [error, setError] = useState<string | null>(null)
  const { theme } = useTheme()

  const getAdvice = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!patientId.trim()) {
      setError('Please enter a patient ID')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      // Speed up the heartbeat during "analysis"
      const heartEl = document.querySelector('.heart') as HTMLElement
      if (heartEl) heartEl.style.animationDuration = "0.8s, 6s"

      const response = await fetch('http://localhost:5001/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ patient_id: patientId })
      })

      const data = await response.json()

      if (response.ok) {
        setResult(data)
        document.getElementById('result')?.classList.add('active')
      } else {
        setError(data.error || 'Error processing request')
      }
    } catch (err) {
      setError('Connection error. Please try again.')
      console.error(err)
    } finally {
      setLoading(false)
      
      // Return heartbeat to normal
      const heartEl = document.querySelector('.heart') as HTMLElement
      if (heartEl) heartEl.style.animationDuration = "2s, 6s"
    }
  }

  const displayResults = () => {
    if (!result) return null

    const riskLevel = result.risk_assessment.risk_level
    const riskScore = (result.risk_assessment.risk_score * 100).toFixed(1)
    
    return (
      <div id="result" className="active">
        <div className="stat-card">
          <div className="stat-title">
            <i className="fas fa-triangle-exclamation title-icon"></i>RISK ASSESSMENT
          </div>
          <div className={`stat-value ${riskLevel === 'High Risk' ? 'risk-high' : 'risk-low'}`}>
            <span>{riskLevel}</span>
            <span>({riskScore}%)</span>
          </div>
          <div className="stat-value" style={{ fontStyle: 'italic', color: theme === 'light' ? '#555' : '#ccc' }}>
            {result.risk_assessment.interpretation}
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-title">
            <i className="fas fa-notes-medical title-icon"></i>PATIENT DETAILS
          </div>
          <div className="stat-value">
            <span>Age</span>
            <span>{result.vital_statistics.Patient_Demographics.Age}</span>
          </div>
          <div className="stat-value">
            <span>Sex</span>
            <span>{result.vital_statistics.Patient_Demographics.Sex}</span>
          </div>
          <div className="stat-value">
            <span>Height</span>
            <span>{result.vital_statistics.Patient_Demographics.Height} cm</span>
          </div>
          <div className="stat-value">
            <span>Weight</span>
            <span>{result.vital_statistics.Patient_Demographics.Weight} kg</span>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-title">
            <i className="fas fa-heartbeat title-icon"></i>MURMUR ASSESSMENT
          </div>
          <div className="stat-value">
            <span>Presence</span>
            <span>{result.vital_statistics.Cardiovascular_Metrics.Murmur_Assessment.Presence ? 'Present' : 'Absent'}</span>
          </div>
          {result.vital_statistics.Cardiovascular_Metrics.Murmur_Assessment.Presence && (
            <>
              <div className="stat-value">
                <span>Location</span>
                <span>{result.vital_statistics.Cardiovascular_Metrics.Murmur_Assessment.Location}</span>
              </div>
              <div className="stat-value">
                <span>Most Audible</span>
                <span>{result.vital_statistics.Cardiovascular_Metrics.Murmur_Assessment.Most_Audible_Location}</span>
              </div>
            </>
          )}
        </div>

        <div className="stat-card">
          <div className="stat-title">
            <i className="fas fa-list-check title-icon"></i>RISK FACTORS
          </div>
          <ul className="insight-list">
            {Array.isArray(result.detailed_analysis.risk_factors) 
              ? result.detailed_analysis.risk_factors.map((factor: string, index: number) => (
                  <li key={index} className="category-header">{factor}</li>
                ))
              : typeof result.detailed_analysis.risk_factors === 'object' && result.detailed_analysis.risk_factors !== null
                ? (
                  <>
                    {result.detailed_analysis.risk_factors.age_specific_factors && (
                      <>
                        <li className="category-header">Age Specific Factors:</li>
                        {typeof result.detailed_analysis.risk_factors.age_specific_factors === 'string' 
                          ? tryParseJsonArray(result.detailed_analysis.risk_factors.age_specific_factors).map((item: string, idx: number) => (
                              <li key={`age-factor-${idx}`} className="sub-item">{item}</li>
                            ))
                          : Array.isArray(result.detailed_analysis.risk_factors.age_specific_factors)
                            ? result.detailed_analysis.risk_factors.age_specific_factors.map((item: string, idx: number) => (
                                <li key={`age-factor-${idx}`} className="sub-item">{item}</li>
                              ))
                            : <li className="sub-item">{JSON.stringify(result.detailed_analysis.risk_factors.age_specific_factors)}</li>
                        }
                      </>
                    )}
                    {result.detailed_analysis.risk_factors.general_risk_factors && (
                      <>
                        <li className="category-header">General Risk Factors:</li>
                        {typeof result.detailed_analysis.risk_factors.general_risk_factors === 'string' 
                          ? tryParseJsonArray(result.detailed_analysis.risk_factors.general_risk_factors).map((item: string, idx: number) => (
                              <li key={`gen-factor-${idx}`} className="sub-item">{item}</li>
                            ))
                          : Array.isArray(result.detailed_analysis.risk_factors.general_risk_factors)
                            ? result.detailed_analysis.risk_factors.general_risk_factors.map((item: string, idx: number) => (
                                <li key={`gen-factor-${idx}`} className="sub-item">{item}</li>
                              ))
                            : <li className="sub-item">{JSON.stringify(result.detailed_analysis.risk_factors.general_risk_factors)}</li>
                        }
                      </>
                    )}
                  </>
                )
                : typeof result.detailed_analysis.risk_factors === 'string' 
                  ? tryParseJsonArray(result.detailed_analysis.risk_factors).map((item: string, idx: number) => (
                      <li key={`risk-factor-${idx}`} className="category-header">{item}</li>
                    ))
                  : <li>No risk factors available</li>
            }
          </ul>
        </div>

        <div className="stat-card warning-card">
          <div className="stat-title">
            <i className="fas fa-exclamation-circle title-icon"></i>WARNING SIGNS
          </div>
          <ul className="warning-list">
            {Array.isArray(result.detailed_analysis.warning_signs)
              ? result.detailed_analysis.warning_signs.map((warning: string, index: number) => (
                  <li key={index} className="category-header">{warning}</li>
                ))
              : typeof result.detailed_analysis.warning_signs === 'object' && result.detailed_analysis.warning_signs !== null
                ? Object.entries(result.detailed_analysis.warning_signs).map(([key, value]) => (
                    <li key={key} className="category-header">{formatKeyName(key)}:
                      {typeof value === 'string' 
                        ? <ul className="nested-list">
                            {tryParseJsonArray(value).map((item: string, idx: number) => (
                              <li key={`${key}-${idx}`} className="sub-item">{item}</li>
                            ))}
                          </ul>
                        : Array.isArray(value)
                          ? <ul className="nested-list">
                              {value.map((item: string, idx: number) => (
                                <li key={`${key}-${idx}`} className="sub-item">{item}</li>
                              ))}
                            </ul>
                          : <span className="value-text"> {JSON.stringify(value)}</span>
                      }
                    </li>
                  ))
                : typeof result.detailed_analysis.warning_signs === 'string' 
                  ? tryParseJsonArray(result.detailed_analysis.warning_signs).map((item: string, idx: number) => (
                      <li key={`warning-${idx}`} className="category-header">{item}</li>
                    ))
                  : <li>No warning signs available</li>
            }
          </ul>
        </div>

        <div className="stat-card positive-card">
          <div className="stat-title">
            <i className="fas fa-check-circle title-icon"></i>POSITIVE INDICATORS
          </div>
          <ul className="positive-list">
            {Array.isArray(result.detailed_analysis.positive_indicators)
              ? result.detailed_analysis.positive_indicators
                  .filter((indicator: string | null) => indicator !== null)
                  .map((indicator: string, index: number) => (
                    <li key={index} className="category-header">{indicator}</li>
                  ))
              : typeof result.detailed_analysis.positive_indicators === 'object' && result.detailed_analysis.positive_indicators !== null
                ? Object.entries(result.detailed_analysis.positive_indicators).map(([key, value]) => (
                    <li key={key} className="category-header">{formatKeyName(key)}:
                      {typeof value === 'string' 
                        ? <ul className="nested-list">
                            {tryParseJsonArray(value).map((item: string, idx: number) => (
                              <li key={`${key}-${idx}`} className="sub-item">{item}</li>
                            ))}
                          </ul>
                        : Array.isArray(value)
                          ? <ul className="nested-list">
                              {value.map((item: string, idx: number) => (
                                <li key={`${key}-${idx}`} className="sub-item">{item}</li>
                              ))}
                            </ul>
                          : <span className="value-text"> {JSON.stringify(value)}</span>
                      }
                    </li>
                  ))
                : typeof result.detailed_analysis.positive_indicators === 'string' 
                  ? tryParseJsonArray(result.detailed_analysis.positive_indicators).map((item: string, idx: number) => (
                      <li key={`positive-${idx}`} className="category-header">{item}</li>
                    ))
                  : <li>No positive indicators available</li>
            }
          </ul>
        </div>

        <div className="stat-card">
          <div className="stat-title">
            <i className="fas fa-hand-holding-heart title-icon"></i>LIFESTYLE RECOMMENDATIONS
          </div>
          <ul className="recommendation-list">
            {Array.isArray(result.detailed_analysis.lifestyle_recommendations)
              ? result.detailed_analysis.lifestyle_recommendations.map((rec: string, index: number) => (
                  <li key={index} className="category-header">{rec}</li>
                ))
              : typeof result.detailed_analysis.lifestyle_recommendations === 'object' && result.detailed_analysis.lifestyle_recommendations !== null
                ? Object.entries(result.detailed_analysis.lifestyle_recommendations).map(([key, value]) => (
                    <li key={key} className="category-header">{formatKeyName(key)}:
                      {typeof value === 'string' 
                        ? <ul className="nested-list">
                            {tryParseJsonArray(value).map((item: string, idx: number) => (
                              <li key={`${key}-${idx}`} className="sub-item">{item}</li>
                            ))}
                          </ul>
                        : Array.isArray(value)
                          ? <ul className="nested-list">
                              {value.map((item: string, idx: number) => (
                                <li key={`${key}-${idx}`} className="sub-item">{item}</li>
                              ))}
                            </ul>
                          : <span className="value-text"> {JSON.stringify(value)}</span>
                      }
                    </li>
                  ))
                : typeof result.detailed_analysis.lifestyle_recommendations === 'string' 
                  ? tryParseJsonArray(result.detailed_analysis.lifestyle_recommendations).map((item: string, idx: number) => (
                      <li key={`lifestyle-${idx}`} className="category-header">{item}</li>
                    ))
                  : <li>No lifestyle recommendations available</li>
            }
          </ul>
        </div>

        <div className="stat-card">
          <div className="stat-title">
            <i className="fas fa-calendar-check title-icon"></i>FOLLOW-UP RECOMMENDATIONS
          </div>
          <ul className="recommendation-list">
            {Array.isArray(result.detailed_analysis.follow_up_recommendations)
              ? result.detailed_analysis.follow_up_recommendations.map((rec: string, index: number) => (
                  <li key={index} className="category-header">{rec}</li>
                ))
              : typeof result.detailed_analysis.follow_up_recommendations === 'object' && result.detailed_analysis.follow_up_recommendations !== null
                ? Object.entries(result.detailed_analysis.follow_up_recommendations).map(([key, value]) => (
                    <li key={key} className="category-header">{formatKeyName(key)}:
                      {typeof value === 'string' 
                        ? <ul className="nested-list">
                            {tryParseJsonArray(value).map((item: string, idx: number) => (
                              <li key={`${key}-${idx}`} className="sub-item">{item}</li>
                            ))}
                          </ul>
                        : Array.isArray(value)
                          ? <ul className="nested-list">
                              {value.map((item: string, idx: number) => (
                                <li key={`${key}-${idx}`} className="sub-item">{item}</li>
                              ))}
                            </ul>
                          : <span className="value-text"> {JSON.stringify(value)}</span>
                      }
                    </li>
                  ))
                : typeof result.detailed_analysis.follow_up_recommendations === 'string' 
                  ? tryParseJsonArray(result.detailed_analysis.follow_up_recommendations).map((item: string, idx: number) => (
                      <li key={`followup-${idx}`} className="category-header">{item}</li>
                    ))
                  : <li>No follow-up recommendations available</li>
            }
          </ul>
        </div>

        <div className="stat-card">
          <div className="stat-title">
            <i className="fas fa-vials title-icon"></i>SUGGESTED TESTS
          </div>
          <ul className="recommendation-list">
            {Array.isArray(result.detailed_analysis.diagnostic_tests)
              ? result.detailed_analysis.diagnostic_tests.map((test: string, index: number) => (
                  <li key={index} className="category-header">{test}</li>
                ))
              : typeof result.detailed_analysis.diagnostic_tests === 'object' && result.detailed_analysis.diagnostic_tests !== null
                ? Object.entries(result.detailed_analysis.diagnostic_tests).map(([key, value]) => (
                    <li key={key} className="category-header">{formatKeyName(key)}:
                      {typeof value === 'string' 
                        ? <ul className="nested-list">
                            {tryParseJsonArray(value).map((item: string, idx: number) => (
                              <li key={`${key}-${idx}`} className="sub-item">{item}</li>
                            ))}
                          </ul>
                        : Array.isArray(value)
                          ? <ul className="nested-list">
                              {value.map((item: string, idx: number) => (
                                <li key={`${key}-${idx}`} className="sub-item">{item}</li>
                              ))}
                            </ul>
                          : <span className="value-text"> {JSON.stringify(value)}</span>
                      }
                    </li>
                  ))
                : typeof result.detailed_analysis.diagnostic_tests === 'string' 
                  ? tryParseJsonArray(result.detailed_analysis.diagnostic_tests).map((item: string, idx: number) => (
                      <li key={`test-${idx}`} className="category-header">{item}</li>
                    ))
                  : <li>No diagnostic tests available</li>
            }
          </ul>
        </div>
      </div>
    )
  }

  return (
    <main>
      {/* Theme Toggle Button */}
      <ThemeToggle />
      
      {/* Simple Hovering Heart Background */}
      <div className="heart-bg">
        <div className="heart"></div>
      </div>
      
      <div className="container">
        <div className="app-content">
          <div className="card">
            <div className="title-container">
              <h2><i className="fas fa-heartbeat heartbeat-icon"></i>Heart Disease Prediction</h2>
            </div>
            
            <form id="prediction-form" className="text-center mb-4" onSubmit={getAdvice}>
              <div className="mb-4">
                <label htmlFor="patient_id" className="form-label fw-bold mb-2">
                  <i className="fas fa-user-md me-2"></i>
                  Enter Patient ID:
                </label>
                <input 
                  type="text" 
                  id="patient_id" 
                  className="form-control w-75 mx-auto" 
                  placeholder="e.g., 12345" 
                  value={patientId}
                  onChange={(e) => setPatientId(e.target.value)}
                  autoComplete="off"
                />
              </div>
              
              <button type="submit" className="btn btn-primary btn-lg px-5" disabled={loading}>
                <i className="fas fa-stethoscope me-2"></i>Get Advice
              </button>
            </form>
            
            {error && (
              <div className="alert alert-danger" role="alert">
                {error}
              </div>
            )}
            
            <div className="loading" id="loading" style={{ display: loading ? 'block' : 'none' }}>
              <div className="spinner"></div>
              <p className="mt-2">Analyzing patient data...</p>
            </div>
          </div>
          
          {/* Results displayed outside the card */}
          {displayResults()}
        </div>

        <style jsx>{`
          .category-header {
            font-weight: bold;
            margin-top: 12px;
            margin-bottom: 8px;
            color: ${theme === 'light' ? '#444' : '#ddd'};
            list-style-type: none;
            position: relative;
            padding-left: 20px;
          }
          
          .category-header:before {
            content: "•";
            position: absolute;
            left: 0;
            color: ${theme === 'light' ? '#555' : '#aaa'};
          }
          
          .sub-item {
            margin-left: 30px;
            color: ${theme === 'light' ? '#555' : '#ccc'};
            padding: 5px 0;
            list-style-type: none;
            position: relative;
            padding-left: 20px;
          }
          
          .sub-item:before {
            content: "○";
            position: absolute;
            left: 0;
            font-size: 0.8em;
            color: ${theme === 'light' ? '#777' : '#999'};
          }
          
          .nested-list {
            list-style: none;
            padding-left: 0;
            margin-top: 6px;
          }
          
          .value-text {
            color: ${theme === 'light' ? '#555' : '#ccc'};
            font-style: italic;
          }
          
          .insight-list, .warning-list, .positive-list, .recommendation-list {
            list-style: none;
            padding-left: 0;
          }
          
          .insight-list li, .warning-list li, .positive-list li, .recommendation-list li {
            padding: 4px 0;
          }
          
          .stat-card {
            margin-bottom: 20px;
            padding: 16px 20px;
            border-radius: 12px;
          }
          
          .stat-title {
            font-size: 1.1rem;
            font-weight: bold;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid ${theme === 'light' ? 'rgba(0,0,0,0.1)' : 'rgba(255,255,255,0.1)'};
          }
        `}</style>
      </div>
    </main>
  )
}
