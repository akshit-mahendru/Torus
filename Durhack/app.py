from flask import Flask, redirect, url_for, send_file
from views import views

app = Flask(__name__)
app.register_blueprint(views, url_prefix="/")

if __name__ == '__main__':
    app.run(debug=True, port=5040)

