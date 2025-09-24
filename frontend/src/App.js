import React, { useState, useEffect } from 'react';
import io from 'socket.io-client';
import styled, { createGlobalStyle, keyframes } from 'styled-components';
import { motion } from 'framer-motion';
import StatsPanel from './components/StatsPanel';
import ModelStatus from './components/ModelStatus';
import ThreatLog from './components/ThreatLog';

const GlobalStyle = createGlobalStyle`
  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }

  body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: linear-gradient(135deg, #e0f7fa 0%, #80deea 50%, #26c6da 100%);
    color: #00363a;
    min-height: 100vh;
    overflow-x: hidden;
  }

  ::-webkit-scrollbar {
    width: 8px;
  }

  ::-webkit-scrollbar-track {
    background: rgba(0, 54, 58, 0.1);
  }

  ::-webkit-scrollbar-thumb {
    background: rgba(0, 54, 58, 0.4);
    border-radius: 4px;
  }
`;

const pulse = keyframes`
  0% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.7; transform: scale(1.05); }
  100% { opacity: 1; transform: scale(1); }
`;

const AppContainer = styled.div`
  min-height: 100vh;
  padding: 0;
`;

const Header = styled(motion.header)`
  background: rgba(255, 255, 255, 0.75);
  border-bottom: 1px solid #007c91;
  padding: 1rem 2rem;
  position: sticky;
  top: 0;
  z-index: 100;
`;

const HeaderContent = styled.div`
  max-width: 1400px;
  margin: 0 auto;
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const Logo = styled.h1`
  font-size: 2rem;
  font-weight: 700;
  background: linear-gradient(135deg, #006064 0%, #004d40 50%, #00251a 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-right: auto;
`;

const StatusBadge = styled(motion.div)`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  background: rgba(0, 124, 145, 0.2);
  border: 1px solid #007c91;
  border-radius: 20px;
  font-size: 0.9rem;
  font-weight: 500;
  color: #004d40;
`;

const StatusDot = styled.div`
  width: 8px;
  height: 8px;
  background: #007c91;
  border-radius: 50%;
  animation: ${pulse} 2s infinite;
`;

const MainContent = styled(motion.main)`
  max-width: 1400px;
  margin: 0 auto;
  padding: 2rem;
  display: grid;
  grid-template-columns: 1fr 1fr;
  grid-template-rows: auto auto;
  gap: 2rem;

  @media (max-width: 768px) {
    grid-template-columns: 1fr;
    padding: 1rem;
    gap: 1rem;
  }
`;

const ThreatLogContainer = styled.div`
  grid-column: 1 / -1;
`;

function App() {
  const [socket, setSocket] = useState(null);
  const [connected, setConnected] = useState(false);
  const [stats, setStats] = useState({
    total_flows: 0,
    threats_detected: 0,
    threats_blocked: 0,
    detection_rate: '0%',
    uptime: '00:00:00',
  });
  const [threats, setThreats] = useState([]);
  const [modelInfo, setModelInfo] = useState(null);

  useEffect(() => {
    const socketConnection = io(process.env.REACT_APP_BACKEND_URL || 'http://localhost:5002');
    setSocket(socketConnection);

    socketConnection.on('connect', () => {
      setConnected(true);
      console.log('Connected to PRATIRAKSHA-Lite backend');
    });

    socketConnection.on('disconnect', () => {
      setConnected(false);
      console.log('Disconnected from backend');
    });

    socketConnection.on('stats_update', (data) => {
      setStats(data);
    });

    socketConnection.on('new_threat', (threat) => {
      setThreats(prev => [threat, ...prev.slice(0, 49)]);
    });

    socketConnection.on('model_info', (info) => {
      setModelInfo(info);
    });

    return () => {
      socketConnection.disconnect();
    };
  }, []);

  return (
    <>
      <GlobalStyle />
      <AppContainer>
        <Header
          initial={{ y: -100 }}
          animate={{ y: 0 }}
          transition={{ type: "spring", stiffness: 100 }}
        >
          <HeaderContent>
            <Logo>🛡️ PRATIRAKSHA-Lite</Logo>
            <StatusBadge
              animate={{ scale: connected ? 1 : 0.95 }}
              transition={{ duration: 0.3 }}
            >
              <StatusDot />
              {connected ? 'Active & Monitoring' : 'Disconnected'}
            </StatusBadge>
          </HeaderContent>
        </Header>

        <MainContent
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3, duration: 0.8 }}
        >
          <StatsPanel stats={stats} />
          <ModelStatus modelInfo={modelInfo} connected={connected} />
          <ThreatLogContainer>
            <ThreatLog threats={threats} />
          </ThreatLogContainer>
        </MainContent>
      </AppContainer>
    </>
  );
}

export default App;
