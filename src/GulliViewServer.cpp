//
// server.cpp
// ~~~~~~~~~~
//
// Copyright (c) 2003-2008 Christopher M. Kohlhoff (chris at kohlhoff dot com)
// Copyright (c) 2013-2014 Andrew Soderberg-Rivkin <sandrew@student.chalmers.se>
// Copyright (c) 2013-2014 Sanjana Hangal <sanjana@student.chalmers.se>
// Copyright (c) 2014 Thomas Petig <petig@chalmers.se>
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
#include "OpenCVHelper.h"
#include "TagDetector.h"

#include <sys/time.h>

#include <ctime>
#include <iostream>
#include <string>
#include <boost/array.hpp>
#include <boost/asio.hpp>
#include <boost/lexical_cast.hpp>
#include <fstream>
#include "boost/date_time/posix_time/posix_time.hpp"

#include "CameraUtil.h"


using namespace std;
using helper::ImageSource;
using boost::asio::ip::udp;
using boost::posix_time::ptime;
using boost::posix_time::time_duration;
int main()
{
   // Create logfile to be used
   std::ofstream fout("GulliViewLog.txt");
   try {
      boost::asio::io_service io_service;

      udp::socket socket(io_service, udp::endpoint(udp::v4(), 13));
      boost::array<uint8_t, 256> answer;

      boost::array<uint8_t, 256> recv_buf;
      for (;;) {
         udp::endpoint remote_endpoint;

         //boost::system::error_code error;
         socket.receive_from( boost::asio::buffer(recv_buf), remote_endpoint);
         ptime recvTime;
         recvTime = boost::posix_time::microsec_clock::local_time();
         uint32_t index = 0;
         uint32_t type    =  recv_buf[0] << 24 | recv_buf[1] << 16 | recv_buf[2] << 8 | recv_buf[3];
         uint32_t sub_type = recv_buf[4] << 24 | recv_buf[5] << 16 | recv_buf[6] << 8 | recv_buf[7];

         //recv_buf[index-1] = 1; //HACK: prepare for retransmission

         //New position data from the camera
         if (type == 1 and sub_type == 2) {
            uint32_t seq     = recv_buf[8] << 24 | recv_buf[9] << 16 | recv_buf[10] << 8 | recv_buf[11];
            index += 16;
            uint32_t len     = recv_buf[28] << 24 | recv_buf[29] << 16 | recv_buf[30] << 8 | recv_buf[31];
            std::cout << "seq " << seq << " len " << len << std::endl;
            for (size_t i = 0; i < len; ++i) {
               int32_t id= recv_buf[32] << 24 | recv_buf[33] << 16 | recv_buf[34] << 8 | recv_buf[35];
               int32_t x = recv_buf[36] << 24 | recv_buf[37] << 16 | recv_buf[38] << 8 | recv_buf[39];
               int32_t y = recv_buf[40] << 24 | recv_buf[41] << 16 | recv_buf[42] << 8 | recv_buf[43];
               int32_t t = recv_buf[44] << 24 | recv_buf[45] << 16 | recv_buf[46] << 8 | recv_buf[47];
               answer = recv_buf;
               std::cout << "Tag: " << id << " x: " << x << " y: " << y << " heading: " << t << std::endl;
            }
            //Data request from the Gulliver map client
         } else if (type == 1 and sub_type == 1) { //TODO: Correct types
             //Set the type to the response type
             answer[3] = 2;
             answer[7] = 2;
             udp::socket replysocket(io_service, udp::endpoint(udp::v4(), 8989));
             udp::endpoint remote_2 (remote_endpoint.address(),4242);
             replysocket.send_to(boost::asio::buffer(answer), remote_2);
         }
         //std::cout<< recvTime << "\n";
//      std::string data(recv_buf.begin(), recv_buf.end());
//      std::cout << "Data: " << data << "\n";
//      unsigned firstDel = data.find('[');
//      unsigned lastDel = data.find(']');
//      string strNew = data.substr (firstDel+1,(lastDel-firstDel)-1);
         //std::cout << strNew << "\n";
//      ptime startProcTime;
//      startProcTime = boost::lexical_cast<ptime>(strNew);
         //std::cout << startProcTime << "\n";
//      time_duration fullProcessTime = recvTime - startProcTime;
         //std::cout << "Full Proc Time: " <<fullProcessTime << "\n";
         // Time Stamp --- Received
         //std::cout << recv_buf.data() << "\n";
         // Write to logfile and save
//      fout << recv_buf.data() << "" << std::endl;


      }
   } catch (std::exception& e) {
      std::cerr << e.what() << std::endl;
   }
   // Close log file
   fout << "Program closed: " << std::endl;
   fout.close();
   return 0;
}
