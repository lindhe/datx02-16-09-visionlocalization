//
// server.cpp
// ~~~~~~~~~~
//
// Copyright (c) 2003-2008 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
#include "OpenCVHelper.h"

#include <ctime>
#include <iostream>
#include <string>
#include <boost/array.hpp>
#include <boost/asio.hpp>
#include <fstream>
#include "boost/date_time/posix_time/posix_time.hpp"

using boost::asio::ip::udp;

std::string make_daytime_string()
{
  using namespace std; // For time_t, time and ctime;
  time_t now = time(0);
  return ctime(&now);
}

int main()
{
  // Create logfile to be used
  std::ofstream fout("GulliViewLog.txt");
  try
  {
    boost::asio::io_service io_service;

    udp::socket socket(io_service, udp::endpoint(udp::v4(), 13));
    
    for (;;)
    {
      boost::array<char, 128> recv_buf;
      udp::endpoint remote_endpoint;
      //boost::system::error_code error;
      socket.receive_from(
        boost::asio::buffer(recv_buf), remote_endpoint);
      //std::cout << recv_buf.data() << "\n";
      // Write to logfile and save
      fout << recv_buf.data() << "" << std::endl;
      
    }
  }
  catch (std::exception& e)
  {
    std::cerr << e.what() << std::endl;
  }
  // Close log file
  fout << "Program closed: " << std::endl;
  fout.close();
  return 0;
}
