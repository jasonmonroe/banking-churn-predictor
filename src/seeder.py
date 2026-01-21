import names
import pycountry
import random
import src.constants as const

class Seeder:
    def __init__(self, count: int = 0):
        if count <= 0:
            count = random.randint(30, const.MAX_DATA_ROWS)

        self.count = count
        self.countries = [country.name for country in pycountry.objects]

    def seed_data(self):
        csv_rows = []
        csv_header = 'row_number,customer_id,surname,credit_score,geography,gender,age,tenure,balance,num_of_products,has_cr_card,is_active_member,estimated_salary,exited'+"\n"
        csv_rows.append(csv_header)

        for i in range(0, self.count):
            csv_row = self.generate_row(i)
            csv_rows.append(csv_row)

        return csv_rows


    def generate_row(self, row_number: int):
        customer_id = self.get_customer_id()
        surname = self.get_surname()
        credit_score = self.get_credit_score()
        geography = self.get_geography()
        gender = self.get_gender()
        age = self.get_age()
        tenure = self.get_tenure()
        balance = self.get_balance()
        num_of_products = self.get_num_of_products()
        has_cr_card = self.get_has_cr_card()
        is_active_member = self.get_is_active_member()
        estimated_salary = self.get_estimated_salary()
        exited = self.get_exited()

        csv_row = f'{row_number},{customer_id},{surname},{credit_score},{geography},{gender},{age},{tenure},{balance},{num_of_products},{has_cr_card},{is_active_member},{estimated_salary},{exited}'+"\n"
        print(f'csv_row = {csv_row}')
        return csv_row

    def get_customer_id(self) -> str:
        return f'{random.randint(0, 99999999):08}'

    def get_surname(self) -> str:
        return names.get_last_name()

    def get_credit_score(self) -> str:
        return str(random.randint(300, 850))

    def get_geography(self) -> str:
        """
        Return to a random country
        :return:
        """
        name = random.choice(self.countries)
        # Check if country name has a space (multi-word)
        if ' ' in name or ',' in name:
            return f'"{name}"'

        return name

    def get_gender(self) -> str:
        if random.randint(1, 100) > 55:
            return 'Male'
        else:
            return 'Female'

    def get_age(self) -> str:
        return str(random.randint(18, 90))

    def get_tenure(self) -> str:
        return str(random.randint(20, 65))

    def get_balance(self) -> str:
        return str(random.randint(0, 160000))

    def get_num_of_products(self) -> str:
        return str(random.randint(0, 8))

    def get_has_cr_card(self) -> str:
        return str(random.randint(0, 1))

    def get_is_active_member(self) -> str:
        if random.randint(1, 100) > 65:
            return str(1)
        else:
            return str(0)

    def get_estimated_salary(self) -> str:
        return str(random.randint(0, 180000))

    def get_exited(self) -> str:
        if random.randint(1, 100) > 70:
            return str(0)
        else:
            return str(1)
