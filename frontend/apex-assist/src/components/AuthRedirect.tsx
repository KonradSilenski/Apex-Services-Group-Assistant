import { useEffect, useState } from 'react';
import { Navigate } from 'react-router';

const PrivateRoute = ({ children }: { children: JSX.Element }) => {
  const [loading, setLoading] = useState(true);
  const [isValid, setIsValid] = useState(false);

  useEffect(() => {
    const verifyToken = async () => {
      const token = localStorage.getItem('token');      

      try {
        console.log('message')
        const response = await fetch(`${import.meta.env.VITE_APP_API_BASE_URL}/auth/verify`, {
          method: 'GET',
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });

        if (response.ok) {
          setIsValid(true);
        } else {
          setIsValid(false);
          localStorage.removeItem('token');
        }
      } catch (error) {
        console.error('Token verification failed:', error);
        setIsValid(false);
        localStorage.removeItem('token');
      } finally {
        setLoading(false);
      }
    };

    verifyToken();
  }, []);

  if (loading) {
    return null;
  }

  return isValid ? children : <Navigate to="/auth/login" replace />;
};

export default PrivateRoute;
