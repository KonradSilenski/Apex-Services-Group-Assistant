import { Button, Label, TextInput } from "flowbite-react";
import { useNavigate } from "react-router";
import { useState } from 'react'


const AuthLogin = () => {
  const navigate = useNavigate();
  const [authing, setAuthing] = useState(false);
  const [username, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  const signIn = async () => {
    setAuthing(true);
    setError('');

    try {
        const formData = new URLSearchParams();
        formData.append('username', username);
        formData.append('password', password);

        const response = await fetch(`${import.meta.env.VITE_APP_API_BASE_URL}/auth/token`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: formData.toString(),
        });

        if (!response.ok) {
            const errorData = await response.json();
            setError(errorData.detail || 'Login failed');
            setAuthing(false);
            return;
        }

        const data = await response.json();
        localStorage.setItem('token', data.access_token);
        console.log('navigate')
        navigate('/');
    } catch (err) {
        console.error(err);
        setError('Something went wrong');
    } finally {
        setAuthing(false);
    }
};
  return (
    <>
      <form >
        <div className="mb-4">
          <div className="mb-2 block">
            <Label htmlFor="email" value="Email" className="text-white"/>
          </div>
          <div className="bg-white rounded-xl overflow-hidden">
          <TextInput
            id="email"
            type="text"
            sizing="md"
            required
            className="bg-white text-black form-control form-rounded-xl"
            onChange={(e) => setEmail(e.target.value)}
          />
          </div>
        </div>
        <div className="mb-4">
          <div className="mb-2 block">
            <Label htmlFor="userpwd" value="Password" className="text-white"/>
          </div>
          <div className="bg-white rounded-xl overflow-hidden">
          <TextInput
            id="userpwd"
            type="password"
            sizing="md"
            required
            className="bg-white text-black form-control form-rounded-xl"
            onChange={(e) => setPassword(e.target.value)} 
            
          />
          </div>
        </div>
        <Button type="submit" onClick={signIn} disabled={authing} className="w-full bg-[#3731af] text-white rounded-xl">
          Sign in
        </Button>
      </form>
      {error && <div className='w-full flex items-center justify-center relative py-4'>{error} </div>}
    </>
  );
};

export default AuthLogin;
