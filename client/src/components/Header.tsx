"use client";

import Image from "next/image";
import React from "react";

const Header = () => {
  return (
    <div>
      <Image
        className="mx-auto py-4"
        src="/logo-white.png"
        alt="file-upload"
        width="256"
        height="256"
      />
    </div>
  );
};

export default Header;
