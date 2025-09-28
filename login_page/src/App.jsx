<<<<<<< HEAD
import LoginPage from './components/LoginPage'

function App() {
  return (
    <>
      <LoginPage />
    </>
  )
}

export default App
=======
import React, { useState } from 'react';
import LoginPage from './components/LoginPage';
import Dashboard from './components/Dashboard';

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [userData, setUserData] = useState(null);

  const handleLoginSuccess = (data) => {
    setUserData(data);
    setIsLoggedIn(true);
  };

  const handleSignOut = () => {
    setIsLoggedIn(false);
    setUserData(null);
  };

  return (
    <>
      {isLoggedIn ? (
        <Dashboard onSignOut={handleSignOut} userData={userData} />
      ) : (
        <LoginPage onLoginSuccess={handleLoginSuccess} />
      )}
    </>
  );
}

export default App;
>>>>>>> master
