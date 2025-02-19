import json
import numpy as np
import tensorflow as tf
import os
import shutil

# 1. データローディング (IDS.jsonから)
def load_data(filepath="IDS.json"):
    """JSONファイルからデータをロードします."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            X = np.array(data['X']).T  # Xを転置
            Y = np.array(data['Y']).T  # Yを転置

            # データ正規化 (例)
            X = (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True) # 標準化
            Y = (Y - np.mean(Y, axis=1, keepdims=True)) / np.std(Y, axis=1, keepdims=True)  # 標準化

            return X, Y, data  # dataも返します
    except FileNotFoundError:
        print(f"エラー: ファイル '{filepath}' が見つかりません.")
        return None, None, None
    except json.JSONDecodeError:
        print(f"エラー: ファイル '{filepath}' から JSON をデコードできません. ファイル形式を確認してください.")
        return None, None, None
    except KeyError as e:
        print(f"エラー: キー '{e}' が JSON ファイルにありません. JSON が 'X' と 'Y' の配列を含むことを確認してください.")
        return None, None, None

# データの保存 (IDS.jsonへ)
def save_data(filepath="IDS.json", data=None):
    """更新されたデータをJSONファイルに保存します."""
    if data is None:
        print("エラー: 保存するデータがありません.")
        return

    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)  # 読みやすくするためにインデントをつけて書き込み
        print(f"データは {filepath} に保存されました.")
    except Exception as e:
        print(f"データの保存中にエラーが発生しました: {e}")

# 画面クリア
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

# ヘルプ表示
def print_help():
    print("コマンド一覧:")
    print("help - 指定されたコマンドの説明 (デフォルトはこのリストを表示)")
    print("save - 指定された名前とディレクトリでモデルを保存 (デフォルトは LoView で現在のディレクトリ)")
    print("load - 指定された名前とディレクトリでモデルをロード (デフォルトは LoView で現在のディレクトリ)")
    print("train - 現在のモデルを指定されたエポック数で学習 (デフォルトは 100 エポック)")
    print("use - ニューラルネットワークの使用 (3つの数字を入力すると、モデルが4つの数字を返します)")
    print("show - データセットから、与えられた入力値に対する「正しい」回答を表示 (use と同様)")
    print("add - データセットへの新しいデータの追加 (X が存在しない場合にのみ)")
    print("edit - データセット内の既存のデータの編集")
    print("remove - データセットからのデータの削除")
    print("delete - 保存されたモデルの削除")
    print("quit - 終了")

# モデルの保存
def save_model(model, name="LoView", directory="."):
    filepath = os.path.join(directory, name)
    model.save(filepath)
    print(f"モデルは {filepath} に保存されました.")

# モデルのロード
def load_model(name="LoView", directory="."):
    filepath = os.path.join(directory, name)
    try:
        model = tf.keras.models.load_model(filepath)
        print(f"モデルは {filepath} からロードされました.")
        return model
    except OSError:
        print(f"エラー: モデルを {filepath} からロードできませんでした. 名前とパスを確認してください.")
        return None

# モデルの学習
def train_model(model, X, Y, epochs=100):
    print("学習を開始します...")
    model.fit(X.T, Y.T, epochs=epochs, batch_size=32, verbose=1)
    print("学習が完了しました.")

# モデルの使用
def use_model(model):
    try:
        num1 = float(input("最初の数字を入力してください: "))
        num2 = float(input("2番目の数字を入力してください: "))
        num3 = input("3番目の数字を入力してください (または 'all'): ").lower()

        if num3 == "all":
            third_numbers = np.linspace(1, 10, 10)  # 1から10までの数字を10個生成
            for i, third_number in enumerate(third_numbers):
                new_example = np.array([[num1], [num2], [third_number]])
                new_example = new_example.T
                prediction = model.predict(new_example)
                print(f"3番目の数字 {third_number:.2f} に対する結果: {prediction}")
        else:
            try:
                num3 = float(num3)
                new_example = np.array([[num1], [num2], [num3]])
                new_example = new_example.T
                prediction = model.predict(new_example)
                print(f"結果: {prediction}")
            except ValueError:
                print("エラー: 3番目の数字に有効な数字を入力してください.")

    except ValueError:
        print("エラー: 1番目と2番目の数字に有効な数字を入力してください.")

# データセットからの「正しい」回答の表示
def show_data(X, Y):  # X と Y を引数として渡す
    try:
        num1 = float(input("最初の数字を入力してください: "))
        num2 = float(input("2番目の数字を入力してください: "))
        num3 = input("3番目の数字を入力してください (または 'all'): ").lower()

        if num3 == "all":
            third_numbers = np.linspace(1, 10, 10)  # 1から10までの数字を10個生成
            for i, third_number in enumerate(third_numbers):
                input_values = np.array([num1, num2, third_number]) # 入力値のベクトルを作成
                found = False
                for j in range(X.shape[1]):
                    if np.allclose(X[:, j], input_values): #Xの各列と比較
                        print(f"3番目の数字 {third_number:.2f} に対する正しい回答: {Y[:, j]}")
                        found = True
                        break  # 一致するものが見つかったので、内側のループから抜けます

                if not found:
                    print(f"3番目の数字 {third_number:.2f} に対するデータは、データセットに見つかりませんでした.")

        else:
            try:
                num3 = float(num3)
                input_values = np.array([num1, num2, num3])
                found = False
                for j in range(X.shape[1]):
                     if np.allclose(X[:, j], input_values):
                         print(f"正しい回答: {Y[:, j]}")
                         found = True
                         break
                if not found:
                    print("データはデータセットに見つかりませんでした.")

            except ValueError:
                print("エラー: 3番目の数字に有効な数字を入力してください.")

    except ValueError:
        print("エラー: 1番目と2番目の数字に有効な数字を入力してください.")

def add_data(X, Y, data):
    """データセットに新しいデータを追加します (Xが存在しない場合のみ)."""
    try:
        num1 = float(input("Xの最初の数字を入力してください: "))
        num2 = float(input("Xの2番目の数字を入力してください: "))
        num3_input = input("Xの3番目の数字を入力してください (または 'all'): ").lower()

        if num3_input == "all":
            num3_values = np.linspace(1, 10, 10)  # 3番目の数字の値を10個作成

            # 存在チェック
            for num3 in num3_values:
                input_values = [num1, num2, num3]
                for existing_x in data['X']:
                    if np.allclose(existing_x, input_values):
                        print(f"エラー: データ X: {input_values} はすでに存在します. 重複を追加できません.")
                        return  # 重複が見つかったら関数を終了

            Y_values = []
            for i in range(len(num3_values)): # 各Xに対してYを要求
                 y1 = float(input(f"Yの最初の数字を入力してください (X: {num1}, {num2}, {num3_values[i]}): "))
                 y2 = float(input(f"Yの2番目の数字を入力してください (X: {num1}, {num2}, {num3_values[i]}): "))
                 y3 = float(input(f"Yの3番目の数字を入力してください (X: {num1}, {num2}, {num3_values[i]}): "))
                 y4 = float(input(f"Yの4番目の数字を入力してください (X: {num1}, {num2}, {num3_values[i]}): "))
                 Y_values.append([y1, y2, y3, y4])

            confirmation = input("これらのデータを追加してもよろしいですか? (y/n): ").lower()

            if confirmation == "y":

                for i in range(len(num3_values)):
                    new_x = [num1, num2, num3_values[i]]
                    new_y = Y_values[i]

                    data['X'].append(new_x)
                    data['Y'].append(new_y)

                save_data(data=data)  # 更新されたデータを保存
                print("データが正常に追加されました.")

            else:
                print("データの追加はキャンセルされました.")


        else:

            num3 = float(num3_input)  # "all" でない場合は数字に変換
            input_values = [num1, num2, num3]  # 探しているXを形成

            # 存在チェック
            for existing_x in data['X']:
                if np.allclose(existing_x, input_values):
                    print(f"エラー: データ X: {input_values} はすでに存在します. 重複を追加できません.")
                    return  # 関数を終了

            y1 = float(input("Yの最初の数字を入力してください: "))
            y2 = float(input("Yの2番目の数字を入力してください: "))
            y3 = float(input("Yの3番目の数字を入力してください: "))
            y4 = float(input("Yの4番目の数字を入力してください: "))

            confirmation = input("このデータを追加してもよろしいですか? (y/n): ").lower()

            if confirmation == "y":

                new_x = [num1, num2, num3]
                new_y = [y1, y2, y3, y4]

                data['X'].append(new_x)
                data['Y'].append(new_y)

                save_data(data=data)  # 更新されたデータを保存
                print("データが正常に追加されました.")

            else:
                print("データの追加はキャンセルされました.")

    except ValueError:
        print("エラー: 有効な数字を入力してください.")
    except Exception as e:
        print(f"エラーが発生しました: {e}")



def edit_data(X, Y, data): # dataを引数として追加
    """データセット内の既存のデータを編集します."""
    try:
        num1 = float(input("検索するXの最初の数字を入力してください: "))
        num2 = float(input("検索するXの2番目の数字を入力してください: "))
        num3_input = input("検索するXの3番目の数字を入力してください (または 'all'): ").lower()

        if num3_input == "all":
            third_numbers = np.linspace(1, 10, 10)
            input_values_list = [[num1, num2, num3] for num3 in third_numbers]  # 検索用のリスト

            for input_values in input_values_list:
              found_index = None # インデックス追跡用
              for i in range(len(data['X'])):
                if np.allclose(data['X'][i], input_values):
                  found_index = i
                  break

              if found_index is not None:
                  print(f"X: {data['X'][found_index]}, Y: {data['Y'][found_index]} に一致するものが見つかりました")

                  confirm_edit = input("このデータを編集しますか? (y/n): ").lower()
                  if confirm_edit == "y":

                      y1 = float(input("Yの新しい最初の数字を入力してください: "))
                      y2 = float(input("Yの新しい2番目の数字を入力してください: "))
                      y3 = float(input("Yの新しい3番目の数字を入力してください: "))
                      y4 = float(input("Yの新しい4番目の数字を入力してください: "))
                      new_y = [y1, y2, y3, y4]

                      confirm_final = input("Yをデータ {} で置き換えますか? (y/n):".format(new_y)).lower() # 確認を追加

                      if confirm_final == "y":
                         data['Y'][found_index] = new_y
                         save_data(data=data)
                         print("データは正常に編集されました.")
                      else:
                        print("編集はキャンセルされました")


                  else:
                      print("編集はキャンセルされました.")
              else:
                   print("データはデータセットに見つかりませんでした.")


        else:
            try:
                num3 = float(num3_input)
                input_values = [num1, num2, num3] # リストに変換

                found_index = None  # 見つかったインデックス

                for i in range(len(data['X'])):  # インデックスを反復処理
                  if np.allclose(data['X'][i], input_values):
                    found_index = i # インデックスを記憶
                    break


                if found_index is not None: # 一致するものが見つかった場合
                  print(f"X: {data['X'][found_index]}, Y: {data['Y'][found_index]} に一致するものが見つかりました") # 出力

                  confirm_edit = input("このデータを編集しますか? (y/n): ").lower()
                  if confirm_edit == "y":

                      y1 = float(input("Yの新しい最初の数字を入力してください: "))
                      y2 = float(input("Yの新しい2番目の数字を入力してください: "))
                      y3 = float(input("Yの新しい3番目の数字を入力してください: "))
                      y4 = float(input("Yの新しい4番目の数字を入力してください: "))
                      new_y = [y1, y2, y3, y4]

                      confirm_final = input("Yをデータ {} で置き換えますか? (y/n):".format(new_y)).lower() # 確認を追加

                      if confirm_final == "y":
                         data['Y'][found_index] = new_y
                         save_data(data=data)
                         print("データは正常に編集されました.")
                      else:
                        print("編集はキャンセルされました")


                  else:
                      print("編集はキャンセルされました.")

                else:
                    print("データはデータセットに見つかりませんでした.")


            except ValueError:
                print("エラー: 3番目の数字に有効な数字を入力してください.")

    except ValueError:
        print("エラー: 有効な数字を入力してください.")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

def remove_data(X, Y, data):
    """データセットからデータを削除します."""
    try:
        num1 = float(input("検索するXの最初の数字を入力してください: "))
        num2 = float(input("検索するXの2番目の数字を入力してください: "))
        num3_input = input("検索するXの3番目の数字を入力してください (または 'all'): ").lower()

        if num3_input == "all":
            third_numbers = np.linspace(1, 10, 10)
            input_values_list = [[num1, num2, num3] for num3 in third_numbers]

            indices_to_remove = []
            for input_values in input_values_list:
                for i in range(len(data['X'])):
                    if np.allclose(data['X'][i], input_values):
                        indices_to_remove.append(i)

            if indices_to_remove:
                print("次の一致するものが見つかりました:")
                for index in indices_to_remove:
                    print(f"X: {data['X'][index]}, Y: {data['Y'][index]}")

                confirmation = input("これらのデータを削除してもよろしいですか? (y/n): ").lower()
                if confirmation == "y":
                    # 削除時にインデックスが壊れないように、最後から要素を削除します
                    for index in sorted(indices_to_remove, reverse=True):
                        del data['X'][index]
                        del data['Y'][index]
                    save_data(data=data)
                    print("データは正常に削除されました.")
                else:
                    print("データの削除はキャンセルされました.")
            else:
                print("データはデータセットに見つかりませんでした.")

        else:
            try:
                num3 = float(num3_input)
                input_values = [num1, num2, num3]

                found_index = None
                for i in range(len(data['X'])):
                    if np.allclose(data['X'][i], input_values):
                        found_index = i
                        break

                if found_index is not None:
                    print(f"一致するものが見つかりました: X: {data['X'][found_index]}, Y: {data['Y'][found_index]}")
                    confirmation = input("これらのデータを削除してもよろしいですか? (y/n): ").lower()
                    if confirmation == "y":
                        del data['X'][found_index]
                        del data['Y'][found_index]
                        save_data(data=data)
                        print("データは正常に削除されました.")
                    else:
                        print("データの削除はキャンセルされました.")
                else:
                    print("データはデータセットに見つかりませんでした.")

            except ValueError:
                print("エラー: 3番目の数字に有効な数字を入力してください.")

    except ValueError:
        print("エラー: 有効な数字を入力してください.")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

def delete_model(directory="."):
    """保存されたモデルを削除します."""
    name = input("削除するモデルの名前を入力してください (デフォルトは LoView): ") or "LoView"
    filepath = os.path.join(directory, name)
    try:
        shutil.rmtree(filepath)  # ディレクトリとそのコンテンツを削除するには、rmtree を使用します
        print(f"モデル {name} は {directory} から正常に削除されました")
    except FileNotFoundError:
        print(f"エラー: モデル {name} は {directory} に見つかりませんでした")
    except OSError as e:
        print(f"モデルの削除中にエラーが発生しました: {e}")



# 2. メインコード
if __name__ == '__main__':
    # レイヤーサイズを設定
    n_input = 3  # 3つの入力セル
    n_hidden = 30000
    n_output = 4  # 4つの出力セル

    # データの読み込み
    X, Y, data = load_data()

    if X is None or Y is None or data is None:
        exit() # データの読み込みに失敗した場合はプログラムを終了

    # データサイズチェック
    if X.shape[0] != n_input:
        raise ValueError(f"入力データのサイズ (X.shape[0]={X.shape[0]}) が予想されるサイズ (n_input={n_input}) と一致しません.")
    if Y.shape[0] != n_output:
        raise ValueError(f"出力データのサイズ (Y.shape[0]={Y.shape[0]}) が予想されるサイズ (n_output={n_output}) と一致しません.")

    # 3. TensorFlow モデルの作成
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(n_hidden, activation='relu', input_shape=(n_input,)),
        tf.keras.layers.Dense(n_output)  # 回帰用の線形アクティベーション
    ])

    # 4. モデルのコンパイル
    model.compile(optimizer='adam', loss='mse')

    # あいさつ
    print("ようこそ!")
    print_help()

    # コマンド処理ループ
    while True:
        command = input("コマンドを入力してください (コマンド一覧を表示するには help): ").lower()
        clear_screen()

        if command == "help":
            print_help()
        elif command == "save":
            name = input("モデルの名前を入力してください (デフォルトは LoView): ") or "LoView"
            directory = input("保存先ディレクトリを入力してください (デフォルトは現在): ") or "."
            save_model(model, name, directory)
        elif command == "load":
            name = input("モデルの名前を入力してください (デフォルトは LoView): ") or "LoView"
            directory = input("ロード元ディレクトリを入力してください (デフォルトは現在): ") or "."
            loaded_model = load_model(name, directory)
            if loaded_model:
                model = loaded_model # 現在のモデルをロードされたモデルに置き換えます
        elif command == "train":
            try:
                epochs = int(input("エポック数を入力してください (デフォルトは 100): ") or 100)
                train_model(model, X, Y, epochs)
            except ValueError:
                print("エラー: エポック数には整数を入力してください.")
        elif command == "use":
            if model is not None:
                use_model(model)
            else:
                print("エラー: モデルがロードされていません. 'load' コマンドを使用してモデルをロードしてください.")
        elif command == "show":
             show_data(X, Y)  # X と Y を show_data 関数に渡します
        elif command == "add":
             add_data(X, Y, data)
             # データの追加後、X と Y を更新する必要があります
             X, Y, data = load_data() # X, Y, dataを再ロードします
        elif command == "edit":
            edit_data(X, Y, data)
            # データの編集後、X と Y を更新する必要があります
            X, Y, data = load_data()  # X, Y, dataを再ロードします
        elif command == "remove":
            remove_data(X, Y, data)
            # データの削除後、X と Y を更新する必要があります
            X, Y, data = load_data()  # X, Y, dataを再ロードします
        elif command == "delete":
            delete_model()
        elif command == "quit":
            print("プログラムを終了します.")
            break
        else:
            print("不明なコマンドです. コマンド一覧を表示するには 'help' を使用してください.")
