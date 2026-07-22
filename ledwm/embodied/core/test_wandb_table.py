import wandb as wb

wb.init(project="test_wandb_table")
table = wb.Table(columns=["a", "b", "c"])
table.add_data(1, 2, 3)
table.add_data(4, 5, 6)
wb.log({"table": table})
