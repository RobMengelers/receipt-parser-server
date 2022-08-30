.PHONY serve:

generate_cert:
	bash util/generate_certificate.sh

generate_token:
	bash util/generate_token.sh

serve:
	bash util/generate_certificate.sh
	bash util/generate_token.sh
	bash util/launcher.sh
	python3 src/receipt_server.py
