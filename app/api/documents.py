from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
)

from app.services.chunking_service import (
    create_chunks_from_extracted_json,
    save_chunks,
)
from app.services.document_registry import (
    get_document,
    register_document,
)
from app.services.embedding_service import (
    create_embeddings_for_chunks,
    save_embeddings,
)
from app.services.faiss_service import (
    create_faiss_index,
    save_faiss_index,
)
from app.services.pdf_service import (
    extract_text_by_page,
    save_extracted_text,
    save_uploaded_pdf,
)


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


@router.post("/upload")
def upload_pdf(
    file: UploadFile = File(...),
):
    """
    Learning/debug endpoint.

    Upload a PDF and extract its text only.
    """
    try:
        pdf_path = save_uploaded_pdf(
            file
        )

        pages = extract_text_by_page(
            pdf_path
        )

        extracted_path = save_extracted_text(
            pdf_path,
            pages,
        )

        pages_with_text = sum(
            1
            for page in pages
            if page.get(
                "text",
                "",
            ).strip()
        )

        return {
            "message": (
                "PDF uploaded and text "
                "extracted successfully"
            ),
            "file_name": pdf_path.name,
            "pages_count": len(pages),
            "pages_with_text": pages_with_text,
            "extracted_file": str(
                extracted_path
            ),
            "preview": (
                pages[0]["text"][:500]
                if pages
                else ""
            ),
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                f"PDF processing failed: "
                f"{error}"
            ),
        ) from error


@router.post("/process")
def process_document(
    file: UploadFile = File(...),
):
    """
    Production-style ingestion endpoint.

    Complete pipeline:
    PDF
    -> extraction/OCR
    -> chunks
    -> embeddings
    -> FAISS
    -> document ID
    """
    try:
        # 1. Save PDF
        pdf_path = save_uploaded_pdf(
            file
        )

        # 2. Extract text / OCR
        pages = extract_text_by_page(
            pdf_path
        )

        extracted_path = save_extracted_text(
            pdf_path,
            pages,
        )

        # 3. Create chunks
        chunks = (
            create_chunks_from_extracted_json(
                extracted_path
            )
        )

        if not chunks:
            raise ValueError(
                "No usable text chunks were "
                "created from the document."
            )

        chunks_path = save_chunks(
            extracted_path,
            chunks,
        )

        # 4. Create embeddings
        embeddings = (
            create_embeddings_for_chunks(
                chunks
            )
        )

        embeddings_path = save_embeddings(
            chunks_path,
            embeddings,
        )

        # 5. Build FAISS index
        index = create_faiss_index(
            embeddings
        )

        index_path = save_faiss_index(
            index,
            chunks_path,
        )

        # 6. Register document
        document_id = register_document(
            original_file_name=file.filename,
            extracted_file_path=str(
                extracted_path
            ),
            chunks_file_path=str(
                chunks_path
            ),
            embeddings_file_path=str(
                embeddings_path
            ),
            faiss_index_path=str(
                index_path
            ),
        )

        return {
            "message": (
                "Document processed successfully"
            ),
            "document_id": document_id,
            "original_file_name": (
                file.filename
            ),
            "pages_count": len(pages),
            "chunks_count": len(chunks),
            "embedding_dimension": (
                embeddings.shape[1]
            ),
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Document processing failed: "
                f"{error}"
            ),
        ) from error


@router.get("/{document_id}")
def get_document_info(
    document_id: str,
):
    """
    Return metadata for a registered document.
    """
    document = get_document(
        document_id
    )

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    return document