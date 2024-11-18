#include <Servo.h>

Servo motor1; // Servo object for motor 1
Servo motor2; // Servo object for motor 2

void setup() {
  motor1.attach(9); // Attach motor 1 to pin 9
  motor2.attach(10); // Attach motor 2 to pin 10
  Serial.begin(9600); // Start serial communication at 9600 baud rate
  Serial.println("System initialized. Waiting for commands...");
}

void loop() {
  if (Serial.available() > 0) { // Check if there is data available to read
    String command = Serial.readStringUntil('\n'); // Read the command
    Serial.print("Received command: "); 
    Serial.println(command); // Print the received command

    if (command == "soft") {
      Serial.println("Executing soft command...");
      motor1.write(90); // Rotate motor 1 to 90 degrees
      delay(1000); // Keep the position for 1 second
      motor1.write(0); // Return motor 1 to 0 degrees
      Serial.println("Soft command executed.");
    } 
    else if (command == "hard") {
      Serial.println("Executing hard command...");
      motor2.write(90); // Rotate motor 2 to 90 degrees
      delay(1000); // Keep the position for 1 second
      motor2.write(0); // Return motor 2 to 0 degrees
      Serial.println("Hard command executed.");
    } 
    else {
      Serial.println("Unknown command. Please send 'soft' or 'hard'.");
    }
  }
}
