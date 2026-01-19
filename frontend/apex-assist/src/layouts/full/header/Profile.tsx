
import { Button, Dropdown } from "flowbite-react";
import { Icon } from "@iconify/react";
import { useNavigate } from "react-router";
import { Link } from "react-router";
import { useState } from "react";
import { useEffect } from "react";

const Profile = () => {
  const navigate = useNavigate();

  const [email, setEmail] = useState<string | null>(null);

  useEffect(() => {
    const fetchUser = async () => {
      const token = localStorage.getItem("token");
      if (!token) {
        console.error("No token found");
        return;
      }

      try {
        const response = await fetch(
          `${import.meta.env.VITE_APP_API_BASE_URL}/users/me`,
          {
            headers: {
              Authorization: `Bearer ${token}`,
              "Content-Type": "application/json",
            },
          }
        );

        if (!response.ok) {
          console.error("Failed to fetch user:", response.status);
          return;
        }

        const data = await response.json();
        setEmail(data.email);
      } catch (err) {
        console.error("Fetch error:", err);
      }
    };

    fetchUser();
  }, []);

  const firstLetter = email ? email.charAt(0).toUpperCase() : "";

  const handleLogout = () => {
    localStorage.removeItem("token");
    navigate("/auth/login");
  };


  return (
    <div className="relative group/menu">
      <Dropdown
        label=""
        className="rounded-sm w-44"
        dismissOnClick={false}
        renderTrigger={() => (
          <span className="h-10 w-10 hover:text-primary hover:bg-lightprimary rounded-full flex justify-center items-center cursor-pointer group-hover/menu:bg-lightprimary group-hover/menu:text-primary">
            <div
              className="w-40 h-10 rounded-full 
                inline-flex items-center justify-center 
                bg-blue-500 hover:bg-blue-200 text-white text-xl font-bold">
              {firstLetter || "?"}
            </div>
          </span>
        )}
      >

        <Dropdown.Item
          as={Link}
          to="#"
          className="px-3 py-3 flex items-center bg-hover group/link w-full gap-3 text-dark"
        >
          <Icon icon="solar:user-circle-outline" height={20} />
          My Profile
        </Dropdown.Item>
        <Dropdown.Item
          as={Link}
          to="#"
          className="px-3 py-3 flex items-center bg-hover group/link w-full gap-3 text-dark"
        >
          <Icon icon="solar:letter-linear" height={20} />
          My Account
        </Dropdown.Item>
        <Dropdown.Item
          as={Link}
          to="#"
          className="px-3 py-3 flex items-center bg-hover group/link w-full gap-3 text-dark"
        >
          <Icon icon="solar:checklist-linear" height={20} />
          My Task
        </Dropdown.Item>
        <div className="p-3 pt-0 flex justify-center">
          <Button
            onClick={handleLogout}
            size="sm"
            className="mt-2 border border-primary text-primary bg-transparent hover:bg-lightprimary outline-none focus:outline-none px-4 py-2 text-sm h-8"
          >
            Logout
          </Button>
        </div>
      </Dropdown>
    </div>
  );
};

export default Profile;
