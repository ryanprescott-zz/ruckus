import { useState } from 'react'
import './App.css'
import AgentsTab from './components/AgentsTab'
import SettingsMenu from './components/SettingsMenu'
import { ThemeProvider } from './contexts/ThemeContext'

function App() {
  const [activeTab, setActiveTab] = useState('agents')

  const tabs = [
    { id: 'agents', label: 'Agents' },
    { id: 'experiments', label: 'Experiments' },
    { id: 'jobs', label: 'Jobs' },
    { id: 'results', label: 'Results' }
  ]

  const renderTabContent = () => {
    switch (activeTab) {
      case 'agents':
        return <AgentsTab />
      case 'experiments':
        return <div className="tab-placeholder">Experiments tab coming soon</div>
      case 'jobs':
        return <div className="tab-placeholder">Jobs tab coming soon</div>
      case 'results':
        return <div className="tab-placeholder">Results tab coming soon</div>
      default:
        return <div className="tab-placeholder">Tab not found</div>
    }
  }

  return (
    <ThemeProvider>
      <div className="app">
        <header className="app-header">
          <div className="header-content">
            <div className="header-title">RUCKUS</div>
            <div className="header-subtitle">MODEL BENCHMARKING PLATFORM</div>
          </div>
          <div className="header-actions">
            <SettingsMenu />
          </div>
        </header>
      <div className="app-content">
        <nav className="tab-navigation">
          {tabs.map(tab => (
            <button
              key={tab.id}
              className={`tab-button ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </nav>
        <div className="tab-content">
          {renderTabContent()}
        </div>
      </div>
    </div>
    </ThemeProvider>
  )
}

export default App