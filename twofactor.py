from flask import Flask, request, redirect, url_for, render_template, session, flash
from twofactor import Client
import random
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for session handling

# Twilio configuration - replace these with your actual credentials
TWILIO_ACCOUNT_SID = 'your_account_sid'
TWILIO_AUTH_TOKEN = 'your_auth_token'
TWILIO_PHONE_NUMBER = 'your_twilio_phone_number'

# Twilio client setup
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

@app.route('/send_verification_code', methods=['GET'])
def send_verification_code():
    # Assuming you have a user's phone number stored or received
    user_phone_number = request.args.get('phone_number')

    if not user_phone_number:
        flash("Phone number is required", "error")
        return redirect(url_for('index'))

    # Generate a 6-digit verification code
    verification_code = str(random.randint(100000, 999999))

    # Store the verification code in the session (for demo purposes)
    session['verification_code'] = verification_code

    # Send the verification code via SMS
    try:
        message = client.messages.create(
            body=f"Your verification code is: {verification_code}",
            from_=TWILIO_PHONE_NUMBER,
            to=user_phone_number
        )
        flash("Verification code sent successfully!", "success")
    except Exception as e:
        flash(f"Failed to send verification code: {str(e)}", "error")
        return redirect(url_for('index'))

    return redirect(url_for('verify_code'))

@app.route('/verify_code', methods=['GET', 'POST'])
def verify_code():
    if request.method == 'POST':
        # Retrieve the code entered by the user
        input_code = request.form.get('verification_code')

        # Check if the entered code matches the one stored in session
        if 'verification_code' in session and input_code == session['verification_code']:
            # Verification successful
            flash("Verification successful!", "success")
            session.pop('verification_code', None)
            return redirect(url_for('index'))
        else:
            # Verification failed
            flash("Invalid verification code. Please try again.", "error")

    return render_template('verify_code.html')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)