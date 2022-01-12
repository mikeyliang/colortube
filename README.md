# Color Tube Sort Game 

## Installation

### Environment

```.sh
$ git clone https://github.com/mikeyliang/colortube
$ cd colortube
$ source ENV_DIR/bin/activate
```
### Packages

```.sh
$ /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
$ brew install tesseract
```

Change game.py for tesseract executable based on PATH

#### Apple M1

```python
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/Cellar/tesseract/5.0.1/bin/tesseract'
```

#### Apple Intel

```python
pytesseract.pytesseract.tesseract_cmd = r'/usr/local/Cellar/tesseract/5.0.1/bin/tesseract'
```

#### Windows (Varies)

```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\tesseract'
```

## License

MIT

---

