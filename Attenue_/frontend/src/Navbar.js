// src/components/Navbar.js
import React, { useState } from "react";
import axios from "axios";
import './Navbar.css';

const Navbar = () => {
    return (
        <>
            <nav className="navbar">
                <h1 id='logo'>Attenue</h1>
                <h1>Attendance Dashboard</h1>
            </nav>
        </>
    );
};

export default Navbar;