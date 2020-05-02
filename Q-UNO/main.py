from Enviroment.UNO_Server import UNOServer

Server = UNOServer({"init_hand_cards": 5, "max_plays": 100})
Clients = Server.init_game()
