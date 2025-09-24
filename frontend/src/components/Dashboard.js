import React from 'react';
import StatsCards from './StatsCards';
import NetworkChart from './NetworkChart';
import ThreatDonut from './ThreatDonut';
// import ThreatLog from './ThreatLog';  // Your previous code
// import ModelStatus from './ModelStatus';

function Dashboard({ stats, chartData, threatTypes, threatFeed }) {
  return (
    <div>
      <StatsCards stats={stats} />
      <div style={{ display: 'flex', gap: '2rem', marginBottom: '2rem' }}>
        <div style={{ flex: 2 }}>
          <NetworkChart data={chartData} />
        </div>
        <div style={{ flex: 1 }}>
          <ThreatDonut data={threatTypes} />
        </div>
      </div>
      {/* <ModelStatus ... /> */}
      {/* <ThreatLog threats={threatFeed} /> */}
    </div>
  );
}

export default Dashboard;
