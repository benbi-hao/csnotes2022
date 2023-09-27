package socket;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.nio.charset.StandardCharsets;

public class UDPSend {

    public static void main(String[] args) {
        DatagramSocket datagramSocket = null;
        try {
            InetAddress id = InetAddress.getByName("192.168.43.112");
            datagramSocket = new DatagramSocket(8081);

            byte[] bytes = "一个简单下消息：\r\n您好朋友".getBytes(StandardCharsets.UTF_8);
            DatagramPacket datagramPacket = new DatagramPacket(bytes, 0, bytes.length, id, 10086);

            datagramSocket.send(datagramPacket);

        } catch (IOException e) {
            throw new RuntimeException(e);
        } finally {
            if (datagramSocket != null) {
                datagramSocket.close();
            }
        }
    }

}
