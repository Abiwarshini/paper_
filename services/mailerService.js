const nodemailer = require('nodemailer');

const sendEmail = async (options) => {
  let transporter;

  if (process.env.EMAIL_HOST && process.env.EMAIL_USER) {
    transporter = nodemailer.createTransport({
      host: process.env.EMAIL_HOST,
      port: process.env.EMAIL_PORT,
      auth: {
        user: process.env.EMAIL_USER,
        pass: process.env.EMAIL_PASS
      }
    });
  } else {
    // Generate test SMTP service from ethereal.email for seamless local testing
    try {
      const testAccount = await nodemailer.createTestAccount();
      transporter = nodemailer.createTransport({
        host: 'smtp.ethereal.email',
        port: 587,
        secure: false,
        auth: {
          user: testAccount.user,
          pass: testAccount.pass
        }
      });
      console.log(`[Mailer] Ethereal SMTP test account initialized: ${testAccount.user}`);
    } catch (err) {
      console.error('Failed to create Ethereal SMTP account, using mock fallback transport.');
      // Fallback mock transporter
      transporter = {
        sendMail: async (msg) => {
          console.log(`[MOCK EMAIL SENT to ${msg.to}] Subject: ${msg.subject}`);
          return { messageId: 'mock-id-12345' };
        }
      };
    }
  }

  const message = {
    from: `${process.env.EMAIL_FROM || 'noreply@growthmonitor.org'}`,
    to: options.email,
    subject: options.subject,
    text: options.message,
    html: options.html || `<p>${options.message}</p>`
  };

  const info = await transporter.sendMail(message);
  console.log(`[Mailer] Email sent successfully: ${info.messageId}`);
  
  if (!process.env.EMAIL_HOST && nodemailer.getTestMessageUrl) {
    const previewUrl = nodemailer.getTestMessageUrl(info);
    if (previewUrl) {
      console.log(`[Mailer] Preview Email at: ${previewUrl}`);
      return previewUrl;
    }
  }
  return null;
};

module.exports = sendEmail;
