from app import app
from models import *
import schedule 

if __name__ == '__main__':
	with app.app_context():
		db.create_all()
		PendingTransactionDatabase.genisis() 
	def run_scheduler():
		while True:
			with app.app_context():
				schedule.run_pending()
				time.sleep(1)  #
	schedule_thread = threading.Thread(target=run_scheduler, daemon=True)
	schedule_thread.start()
	app.run(host="0.0.0.0",port=8080)