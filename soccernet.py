from SoccerNet.Downloader import SoccerNetDownloader

mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="")
mySoccerNetDownloader.downloadDataTask(task="calibration-2023", split=["train", "valid", "test", "challenge"])

# mySoccerNetDownloader.downloadDataTask(task="tracking-2023", split=["train", "test", "challenge"]) # Traacking 
# mySoccerNetDownloader.downloadDataTask(task="tracking", split=["train", "test", "challenge"])# Traacking 
# # Download SoccerNet labels
# mySoccerNetDownloader.downloadGames(files=["Labels.json"], split=["train", "valid", "test"]) # download labels
# mySoccerNetDownloader.downloadGames(files=["Labels-v2.json"], split=["train", "valid", "test"]) # Action spot labels 
# mySoccerNetDownloader.downloadGames(files=["Labels-cameras.json"], split=["train", "valid", "test"]) # Lables for the camera shots 

# Download SoccerNet features
# mySoccerNetDownloader.downloadGames(files=["1_ResNET_TF2.npy", "2_ResNET_TF2.npy"], split=["train", "valid", "test"]) # download Features
# mySoccerNetDownloader.downloadGames(files=["1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy"], split=["train", "valid", "test"]) # download Features reduced with PCA
# mySoccerNetDownloader.downloadGames(files=["1_player_boundingbox_maskrcnn.json", "2_player_boundingbox_maskrcnn.json"], split=["train", "valid", "test"]) # download Player Bounding Boxes inferred with MaskRCNN
# mySoccerNetDownloader.downloadGames(files=["1_field_calib_ccbv.json", "2_field_calib_ccbv.json"], split=["train", "valid", "test"]) # download Field Calibration inferred with CCBV
# mySoccerNetDownloader.downloadGames(files=["1_baidu_soccer_embeddings.npy", "2_baidu_soccer_embeddings.npy"], split=["train", "valid", "test"]) # download Frame Embeddings from https://github.com/baidu-research/vidpress-sports

#  Download SoccerNet Challenge set (require password from NDA to download videos)
# mySoccerNetDownloader.downloadGames(files=["1_ResNET_TF2.npy", "2_ResNET_TF2.npy"], split=["challenge"]) # download ResNET Features
# mySoccerNetDownloader.downloadGames(files=["1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy"], split=["challenge"]) # download ResNET Features reduced with PCA
# mySoccerNetDownloader.downloadGames(files=["1_224p.mkv", "2_224p.mkv"], split=["challenge"]) # download 224p Videos (require password from NDA)
# mySoccerNetDownloader.downloadGames(files=["1_720p.mkv", "2_720p.mkv"], split=["challenge"]) # download 720p Videos (require password from NDA)
# mySoccerNetDownloader.downloadGames(files=["1_player_boundingbox_maskrcnn.json", "2_player_boundingbox_maskrcnn.json"], split=["challenge"]) # download Player Bounding Boxes inferred with MaskRCNN
# mySoccerNetDownloader.downloadGames(files=["1_field_calib_ccbv.json", "2_field_calib_ccbv.json"], split=["challenge"]) # download Field Calibration inferred with CCBV
# mySoccerNetDownloader.downloadGames(files=["1_baidu_soccer_embeddings.npy", "2_baidu_soccer_embeddings.npy"], split=["challenge"]) # download Frame Embeddings from https://github.com/baidu-research/vidpress-sports

# # Download development kit per task
# # mySoccerNetDownloader.downloadDataTask(task="caption-2023", split=["train", "valid", "test", "challenge"])
# # mySoccerNetDownloader.downloadDataTask(task="jersey-2023", split=["train", "test", "challenge"])
# mySoccerNetDownloader.downloadDataTask(task="reid-2023", split=["train", "valid", "test", "challenge"])
# # mySoccerNetDownloader.downloadDataTask(task="spotting-2023", split=["train", "valid", "test", "challenge"])
# # mySoccerNetDownloader.downloadDataTask(task="spotting-ball-2023", split=["train", "valid", "test", "challenge"], password="s0cc3rn3t")
mySoccerNetDownloader.downloadDataTask(task="tracking-2023", split=["train", "test", "challenge"]) # Traacking 
mySoccerNetDownloader.downloadDataTask(task="tracking", split=["train", "test", "challenge"])# Traacking 
# # mySoccerNetDownloader.downloadDataTask(task="SpiideoSynLoc", split=["train","valid","test","challenge"]) # 4K Images
# # mySoccerNetDownloader.downloadDataTask(task="SpiideoSynLoc", split=["train","valid","test","challenge"], version="fullhd") # FullHD Images

# Download SoccerNet videos (require password from NDA to download videos)
# mySoccerNetDownloader.password = "s0cc3rn3t"
# mySoccerNetDownloader.downloadGames(files=["1_224p.mkv", "2_224p.mkv"], split=["train", "valid", "test"]) # download 224p Videos
# mySoccerNetDownloader.downloadGames(files=["1_720p.mkv", "2_720p.mkv"], split=["train", "valid", "test"]) # download 720p Videos
# mySoccerNetDownloader.downloadRAWVideo(dataset="SoccerNet") # download 720p Videos
mySoccerNetDownloader.downloadRAWVideo(dataset="SoccerNet-Tracking") # download single camera RAW Videos

# # Download SoccerNet in OSL ActionSpotting format
# mySoccerNetDownloader.downloadDataTask(task="spotting-OSL", split=["train", "valid", "test", "challenge"], version="ResNET_PCA512")
# mySoccerNetDownloader.downloadDataTask(task="spotting-OSL", split=["train", "valid", "test", "challenge"], version="baidu_soccer_embeddings")
# mySoccerNetDownloader.downloadDataTask(task="spotting-OSL", split=["train", "valid", "test", "challenge"], version="224p", password="s0cc3rn3t")