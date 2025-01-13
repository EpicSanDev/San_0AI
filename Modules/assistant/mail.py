import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class ICloudMailAssistant:
    def __init__(self, email, app_specific_password):
        self.email = email
        self.password = app_specific_password
        self.smtp_server = "smtp.mail.me.com"
        self.port = 587
        
    def send_email(self, to_email, subject, body):
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email
            msg['To'] = to_email
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.smtp_server, self.port)
            server.starttls()
            server.login(self.email, self.password)
            server.send_message(msg)
            server.quit()
            return True
        except Exception as e:
            print(f"Erreur d'envoi d'email: {str(e)}")
            return False

    def connect_icloud(self):
        try:
            self.server = smtplib.SMTP(self.smtp_server, self.port)
            self.server.starttls()
            self.server.login(self.email, self.password)
            return True
        except Exception as e:
            print(f"Erreur de connexion iCloud: {str(e)}")
            return False
