import pandas as pd
import random
from datetime import datetime, timedelta
import uuid
from faker import Faker

fake = Faker()

order_ids = ['ord' + str(i).zfill(6) for i in range(1001, 11001)]

start_date = datetime.now() - timedelta(days=30)
end_date = datetime.now() + timedelta(days=30)

order_dates = [start_date + timedelta(days=random.randint(0, 30)) for _ in range(10000)]
order_dates_formatted = [date.strftime('%Y-%m-%d') for date in order_dates]

estimated_delivery_dates = [order_date + timedelta(days=random.randint(3, 30)) for order_date in order_dates]
estimated_delivery_dates_formatted = [date.strftime('%Y-%m-%d') for date in estimated_delivery_dates]

statuses = ['processing', 'shipped', 'delivered', 'cancelled']

order_statuses = [random.choice(statuses) for _ in range(10000)]

tracking_ids = [uuid.uuid4() for _ in range(10000)]

customer_ids = [uuid.uuid4() for _ in range(10000)]

supplier_ids = [uuid.uuid4() for _ in range(10000)]

customer_contacts = [fake.email() for _ in range(10000)]

supplier_contacts = [fake.email() for _ in range(10000)]


order_data = {
    'order_id': order_ids,
    'order_date': order_dates_formatted,
    'estimated_delivery_date': estimated_delivery_dates_formatted,
    'status': order_statuses,
    'tracking_id': tracking_ids,
    'customer_id': customer_ids,
    'supplier_id': supplier_ids,
    'customer_contact': customer_contacts,
    'supplier_contact': supplier_contacts
}

order_df = pd.DataFrame(order_data)

order_df.to_csv('orders.csv', index=False)

print("Data saved to 'orders.csv'")
