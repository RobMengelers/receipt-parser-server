import util as u


# Send info message
def info(skk): print("\033[95m{}\033[00m".format(skk))


# Send error message
def error(skk): print("\033[91m{}\033[00m".format(skk))


# Print receipt list
def print_receipt_list(receipt):
    print(receipt)
    if not receipt.items:
        return

    for item in receipt.items:
        print("\t*", item.article, " ", item.sum)


# Print full receipt
def print_receipt(receipt):
    print("Company:    ", u.save_ret(receipt.market))
    print("Date:       ", u.save_ret(receipt.date))
    print("Amount:     ", u.save_ret(receipt.sum))
    print("Items:     ")

    print_receipt_list(receipt)
