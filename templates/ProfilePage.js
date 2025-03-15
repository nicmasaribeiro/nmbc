//import React, { useEffect, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Bell, Wallet, Briefcase, PieChart } from "lucide-react";

const ProfilePage = () => {
  const [notifications, setNotifications] = useState([]);
  const [investments, setInvestments] = useState({});
  const [portfolio, setPortfolio] = useState({});
  const [wallet, setWallet] = useState({});

  useEffect(() => {
    fetch("/api/notifications")
      .then((res) => res.json())
      .then((data) => setNotifications(data.notifications));

    fetch("/api/investments")
      .then((res) => res.json())
      .then((data) => setInvestments(data));

    fetch("/api/portfolio")
      .then((res) => res.json())
      .then((data) => setPortfolio(data));

    fetch("/api/wallet")
      .then((res) => res.json())
      .then((data) => setWallet(data));
  }, []);

  return (
    <div className="p-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      <Card className="p-4 flex items-center space-x-4">
        <Bell size={32} className="text-blue-500" />
        <CardContent>
          <h2 className="text-xl font-bold">Notifications</h2>
          <ul>
            {notifications.map((note, index) => (
              <li key={index} className="text-gray-500">{note}</li>
            ))}
          </ul>
        </CardContent>
      </Card>

      <Card className="p-4 flex items-center space-x-4">
        <Briefcase size={32} className="text-green-500" />
        <CardContent>
          <h2 className="text-xl font-bold">Investments</h2>
          <p className="text-gray-500">${investments.total} {investments.currency}</p>
        </CardContent>
      </Card>

      <Card className="p-4 flex items-center space-x-4">
        <PieChart size={32} className="text-purple-500" />
        <CardContent>
          <h2 className="text-xl font-bold">Portfolio</h2>
          <p className="text-gray-500">{portfolio.assets} assets</p>
        </CardContent>
      </Card>

      <Card className="p-4 flex items-center space-x-4">
        <Wallet size={32} className="text-yellow-500" />
        <CardContent>
          <h2 className="text-xl font-bold">Wallet</h2>
          <p className="text-gray-500">Balance: ${wallet.balance} {wallet.currency}</p>
        </CardContent>
      </Card>
    </div>
  );
};

export default ProfilePage;
