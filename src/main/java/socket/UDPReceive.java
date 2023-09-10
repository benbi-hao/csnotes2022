package socket;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.nio.charset.StandardCharsets;

public class UDPReceive {
    public static void main(String[] args) {
        DatagramSocket datagramSocket = null;
        try {
            datagramSocket = new DatagramSocket(10086);

            byte[] bytes = new byte[1024];
            DatagramPacket datagramPacket = new DatagramPacket(bytes, 0, bytes.length);
            datagramSocket.receive(datagramPacket);

            byte[] data = datagramPacket.getData();
            int length = datagramPacket.getLength();
            InetAddress address = datagramPacket.getAddress();
            int port = datagramPacket.getPort();
            String str = new String(data, 0, length, StandardCharsets.UTF_8);
            System.out.println("接收数据:");
            System.out.println(str);
            System.out.println("从哪个ip发送来的数据：" + address + " 对方使用哪个端口发送数据：" + port);

        } catch (IOException e) {
            throw new RuntimeException(e);
        } finally {
            if (datagramSocket != null) {
                datagramSocket.close();
            }
        }
    }
}
