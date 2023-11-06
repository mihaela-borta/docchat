
import modal
import etl.shared

# extend the shared image with PDF-handling dependencies
image = etl.shared.image.pip_install(
    )

stub = modal.Stub(
    name="etl-latex",
    image=image,
    secrets=[
        modal.Secret.from_name("mongodb-fsdl"),
    ],
    mounts=[
        # we make our local modules available to the container
        modal.Mount.from_local_python_packages("docstore", "utils")
    ],
)

'''@stub.local_entrypoint()
def main_1(dir_path="data/latex", collection=None, db=None):
    """Calls the ETL pipeline using a directory containing the Latex files.

    modal run etl/latex.py --dir-path /path/to/latex/documents/dir
    """
    from pathlib import Path
    import datetime

    dir_path = Path(dir_path).resolve()
    if not dir_path.exists():
        print(f"{dir_path} not found, writing to it from the database.")

    title = dir_path.name
    files = [file for file in dir_path.iterdir() if file.suffix == '.mmd']
    documents = []


    for file in sorted(files):
        doc = dict()
        metadata = dict()
        metadata['page'] = file.name.replace('page_', '')
        metadata['is_endmatter'] = False
        metadata['title'] = title
        metadata['date'] = datetime.datetime(2018, 1, 1, 0, 0, 0)
        metadata['full-title'] = title
        #metadata['sha256'] = 'c3654fcd9609c733a87b7bb4997e7a2b8bf83a93e571eeec73915c1aa5eb4c88'
        metadata['ignore'] = False
        metadata['source'] = title
        doc['metadata'] = metadata

        with open(file) as f: text = f.read()
        doc['text'] = text
        documents.append(doc)
    documents = etl.shared.enrich_metadata(documents)

    from pprint import pprint
    print('LATEX DOCUMENT!!!!')
    pprint(documents[0].keys())
    pprint(documents[0]['metadata'].keys())
    #pprint(documents[0])

    with etl.shared.stub.run():
        chunked_documents = etl.shared.chunk_into(documents, 10)
        list(
            etl.shared.add_to_document_db.map(
                chunked_documents, kwargs={"db": db, "collection": collection}
            )
        )
'''


@stub.local_entrypoint()
def main(main_dir_path="data/latex", collection=None, db=None):
    """Calls the ETL pipeline for multiple folders with LaTeX documents.

    modal run etl/latex.py --main_dir_path /path/to/main/latex/documents
    """
    from pathlib import Path
    import datetime

    main_dir_path = Path(main_dir_path).resolve()
    if not main_dir_path.exists():
        print(f"{main_dir_path} not found.")
        return

    subfolders = sorted(main_dir_path.iterdir())

    for subfolder in subfolders:
        title = subfolder.name
        print(f'{title}')
        files = [file for file in subfolder.iterdir() if file.suffix == '.mmd']
        documents = []

        for file in sorted(files):
            doc = dict()
            metadata = dict()
            metadata['page'] = file.name.replace('page_', '')
            metadata['is_endmatter'] = False
            metadata['title'] = title
            metadata['date'] = datetime.datetime(2018, 1, 1, 0, 0, 0)
            metadata['full-title'] = title
            metadata['ignore'] = False
            metadata['source'] = f'{title}: [{metadata["page"]}]'
            doc['metadata'] = metadata

            with open(file) as f:
                text = f.read()
            doc['text'] = text
            documents.append(doc)
        
        documents = etl.shared.enrich_metadata(documents)

        with etl.shared.stub.run():
            chunked_documents = etl.shared.chunk_into(documents, 10)
            list(
                etl.shared.add_to_document_db.map(
                    chunked_documents, kwargs={"db": db, "collection": collection}
                )
            )

