import os


def main(dir='/home/sebastianwidmann/Documents/git/nacaTransformer/airfoilMNIST'):

    for root, dirs, files in os.walk(dir):
        for name in files:
            if name == 'internal.vtu':
                source_path = os.path.join(root, name)
                target_path = os.path.join('/home/sebastianwidmann/Documents/git/nacaTransformer/airfoilMNIST/' + os.path.basename(root) + '.vtu')

                os.system('mv' +' ' + source_path + ' ' + target_path)

if __name__ == '__main__':
    main()