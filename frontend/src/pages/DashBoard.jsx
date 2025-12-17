import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { Zap, Activity, TrendingUp, Calendar, AlertTriangle, DollarSign } from 'lucide-react';
import axios from 'axios';

// --- EMBEDDED CSS ---
const styles = `
  /* Base Styles */
  body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    background-color: #f9fafb;
    color: #1f2937;
  }

  .dashboard-container {
    min-height: 100vh;
    padding: 2rem;
    max-width: 1400px;
    margin: 0 auto;
  }

  /* Header */
  .dashboard-header {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 2rem;
  }

  @media (min-width: 768px) {
    .dashboard-header {
      flex-direction: row;
    }
  }

  .header-title-section {
    text-align: center;
  }

  @media (min-width: 768px) {
    .header-title-section {
      text-align: left;
    }
  }

  .app-title {
    font-size: 1.875rem;
    font-weight: 700;
    color: #111827;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin: 0;
  }

  .title-icon {
    color: #2563eb;
    fill: currentColor;
  }

  .app-subtitle {
    color: #6b7280;
    margin-top: 0.25rem;
    font-size: 1rem;
  }

  /* Controls */
  .controls-section {
    display: flex;
    gap: 1rem;
    background-color: #ffffff;
    padding: 0.5rem;
    border-radius: 0.5rem;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    margin-top: 1rem;
  }

  @media (min-width: 768px) {
    .controls-section {
      margin-top: 0;
    }
  }

  .region-select {
    background-color: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 0.375rem;
    padding: 0.5rem 1rem;
    font-size: 1rem;
    outline: none;
  }

  .region-select:focus {
    border-color: #3b82f6;
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
  }

  .update-button {
    background-color: #2563eb;
    color: white;
    padding: 0.5rem 1.5rem;
    border-radius: 0.375rem;
    font-weight: 500;
    border: none;
    cursor: pointer;
    transition: background-color 0.2s;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .update-button:hover {
    background-color: #1d4ed8;
  }

  .update-button:disabled {
    background-color: #93c5fd;
    cursor: not-allowed;
  }

  /* Error Banner */
  .error-banner {
    background-color: #fef2f2;
    border-left: 4px solid #ef4444;
    padding: 1rem;
    margin-bottom: 1.5rem;
    border-radius: 0 0.375rem 0.375rem 0;
  }

  .error-content {
    display: flex;
    align-items: center;
  }

  .error-icon {
    height: 1.25rem;
    width: 1.25rem;
    color: #ef4444;
    margin-right: 0.5rem;
  }

  .error-content p {
    color: #b91c1c;
    margin: 0;
  }

  /* Stats Grid */
  .stats-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 1.5rem;
    margin-bottom: 2rem;
  }

  @media (min-width: 768px) {
    .stats-grid {
      grid-template-columns: repeat(2, 1fr);
    }
  }

  @media (min-width: 1024px) {
    .stats-grid {
      grid-template-columns: repeat(4, 1fr);
    }
  }

  .stat-card {
    background-color: white;
    padding: 1.5rem;
    border-radius: 0.75rem;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    border: 1px solid #f3f4f6;
    display: flex;
    align-items: center;
    gap: 1rem;
  }

  .stat-icon-wrapper {
    padding: 0.75rem;
    border-radius: 9999px;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .stat-icon {
    width: 1.5rem;
    height: 1.5rem;
  }

  /* Stat Icon Colors */
  .icon-blue { background-color: rgba(59, 130, 246, 0.1); color: #3b82f6; }
  .icon-red { background-color: rgba(239, 68, 68, 0.1); color: #ef4444; }
  .icon-green { background-color: rgba(34, 197, 94, 0.1); color: #22c55e; }
  .icon-purple { background-color: rgba(168, 85, 247, 0.1); color: #a855f7; }
  .icon-orange { background-color: rgba(249, 115, 22, 0.1); color: #f97316; }

  .stat-content p {
    color: #6b7280;
    font-size: 0.875rem;
    font-weight: 500;
    margin: 0;
  }

  .stat-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #1f2937;
    margin: 0;
  }

  /* Charts Layout */
  .charts-layout {
    display: grid;
    grid-template-columns: 1fr;
    gap: 2rem;
  }

  @media (min-width: 1024px) {
    .charts-layout {
      grid-template-columns: 2fr 1fr;
    }
  }

  .chart-card {
    background-color: white;
    padding: 1.5rem;
    border-radius: 0.75rem;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    border: 1px solid #f3f4f6;
  }

  .card-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #1f2937;
    margin-bottom: 1.5rem;
    margin-top: 0;
  }

  .chart-wrapper {
    height: 400px;
    width: 100%;
  }

  .chart-footer {
    font-size: 0.875rem;
    color: #9ca3af;
    margin-top: 1rem;
    text-align: center;
  }

  /* Price List Styling */
  .price-list {
    margin-top: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  .price-header {
    display: flex;
    justify-content: space-between;
    font-size: 0.875rem;
    border-bottom: 1px solid #e5e7eb;
    padding-bottom: 0.5rem;
    color: #6b7280;
  }

  .bold-header {
    font-weight: 600;
    color: #374151;
  }

  .price-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.875rem;
  }

  .price-date {
    color: #6b7280;
  }

  .price-val {
    font-weight: 500;
    color: #ea580c; /* Orange color */
  }
`;

const Dashboard = () => {
  // State
  const [region, setRegion] = useState('Western');
  const [includePrices, setIncludePrices] = useState(true);
  const [loading, setLoading] = useState(false);
  const [gridData, setGridData] = useState([]);
  const [error, setError] = useState(null);

  // Fetch Data
  const fetchForecast = async () => {
    setLoading(true);
    setError(null);
    try {
      // Calling your Flask backend
      const response = await axios.get(`http://127.0.0.1:5000/predict`, {
        params: { region, include_prices: includePrices }
      });
      
      const { history, forecast } = response.data.data;
      
      // Merge history and forecast for the graph
      const mergedData = [
        ...history.map(d => ({ ...d, isForecast: false })),
        ...forecast.map(d => ({ ...d, isForecast: true }))
      ];
      
      setGridData(mergedData);
    } catch (err) {
      console.error(err);
      setError("Failed to fetch data. Is the Flask server running?");
    } finally {
      setLoading(false);
    }
  };

  // Initial Load
  useEffect(() => {
    fetchForecast();
  }, []);

  // --- UI COMPONENTS ---

  const StatCard = ({ title, value, icon: Icon, colorClass }) => (
    <div className="stat-card">
      <div className={`stat-icon-wrapper ${colorClass}`}>
        <Icon className="stat-icon" />
      </div>
      <div className="stat-content">
        <p className="stat-title">{title}</p>
        <h3 className="stat-value">{value}</h3>
      </div>
    </div>
  );

  return (
    <>
      <style>{styles}</style>
      <div className="dashboard-container">
        
        {/* Header */}
        <div className="dashboard-header">
          <div className="header-title-section">
            <h1 className="app-title">
              <Zap className="title-icon" />
              PJM Grid Intelligence
            </h1>
            <p className="app-subtitle">AI-Powered Outage & Price Forecasting System</p>
          </div>
          
          {/* Controls */}
          <div className="controls-section">
            <select 
              value={region} 
              onChange={(e) => setRegion(e.target.value)}
              className="region-select"
            >
              <option value="Western">Western Region</option>
              <option value="PJM RTO">PJM RTO (Entire Grid)</option>
              <option value="Mid Atlantic - Dominion">Mid-Atlantic</option>
            </select>
            
            <button 
              onClick={fetchForecast}
              className="update-button"
              disabled={loading}
            >
              {loading ? 'Analyzing...' : 'Update Forecast'}
            </button>
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <div className="error-banner">
            <div className="error-content">
              <AlertTriangle className="error-icon" />
              <p>{error}</p>
            </div>
          </div>
        )}

        {/* Stats Row */}
        {gridData.length > 0 && (
          <div className="stats-grid">
            <StatCard 
              title="Current Outage" 
              value={`${gridData[gridData.length - 6].outage_mw} MW`} 
              icon={Activity} 
              colorClass="icon-blue" 
            />
            <StatCard 
              title="5-Day Trend" 
              value={gridData[gridData.length - 1].outage_mw > gridData[gridData.length - 6].outage_mw ? 'Increasing ↗' : 'Decreasing ↘'} 
              icon={TrendingUp} 
              colorClass={gridData[gridData.length - 1].outage_mw > gridData[gridData.length - 6].outage_mw ? "icon-red" : "icon-green"} 
            />
            <StatCard 
              title="Forecast Horizon" 
              value="5 Days" 
              icon={Calendar} 
              colorClass="icon-purple" 
            />
            <StatCard 
              title="Avg Predicted Price (Gas)" 
              value={`$${(gridData.slice(-5).reduce((acc, curr) => acc + (curr.predicted_prices?.natural_gas_price || 0), 0) / 5).toFixed(2)}`} 
              icon={DollarSign} 
              colorClass="icon-orange" 
            />
          </div>
        )}

        {/* Main Charts Area */}
        {gridData.length > 0 && (
          <div className="charts-layout">
            
            {/* Outage Trend Chart */}
            <div className="chart-card main-chart">
              <h2 className="card-title">Total Outage Trend (History vs Forecast)</h2>
              <div className="chart-wrapper">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={gridData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                    <defs>
                      <linearGradient id="colorOutage" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.2}/>
                        <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                    <XAxis dataKey="date" tick={{fontSize: 12}} />
                    <YAxis unit=" MW" />
                    <Tooltip 
                      contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }}
                    />
                    <Legend />
                    <Area 
                      type="monotone" 
                      dataKey="outage_mw" 
                      stroke="#3b82f6" 
                      strokeWidth={3}
                      fillOpacity={1} 
                      fill="url(#colorOutage)" 
                      name="Outage (MW)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              <p className="chart-footer">
                The graph transitions from actual historical data to ML-predicted values for the next 5 days.
              </p>
            </div>

            {/* Price Forecast Chart */}
            <div className="chart-card side-chart">
              <h2 className="card-title">Forecasted Fuel Prices</h2>
              <div className="chart-wrapper">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={gridData.filter(d => d.isForecast)}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="date" tick={{fontSize: 10}} />
                    <YAxis prefix="$" width={40} />
                    <Tooltip />
                    <Legend wrapperStyle={{paddingTop: '20px'}}/>
                    
                    <Line type="monotone" dataKey="predicted_prices.natural_gas_price" name="Gas" stroke="#f97316" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="predicted_prices.coal_price" name="Coal" stroke="#4b5563" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="predicted_prices.solar_price" name="Solar" stroke="#eab308" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="predicted_prices.wind_price" name="Wind" stroke="#0ea5e9" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div className="price-list">
                 <div className="price-header">
                    <span>Date</span>
                    <span className="bold-header">Gas Price</span>
                 </div>
                 {gridData.filter(d => d.isForecast).map(day => (
                   <div key={day.date} className="price-row">
                      <span className="price-date">{day.date}</span>
                      <span className="price-val">${day.predicted_prices?.natural_gas_price}</span>
                   </div>
                 ))}
              </div>
            </div>

          </div>
        )}
      </div>
    </>
  );
};

export default Dashboard;