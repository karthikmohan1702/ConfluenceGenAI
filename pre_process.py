import pandas as pd
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_page_ids(confluence_api):
    pages = confluence_api.get_all_pages_from_space(space="chatmodel", content_type='page')
    page_id = [d['id'] for d in pages if 'id' in d]
    return page_id

def extract_page_history(pages:list, confluence_api):
    page_data = []

    for page_id in pages:
        data = confluence_api.history(page_id=int(page_id))
        
        # Extract "created" and "last updated" details
        created_date = data.get('createdDate')
        last_updated_when = data['lastUpdated']['when']

        page_data.append({
            'page_id': page_id,
            'Created': created_date,
            'LastUpdated': last_updated_when
        })

    # Create a Pandas DataFrame
    df = pd.DataFrame(page_data)
    return df

def extract_page_content(page_id, confluence_api):
    html_content = confluence_api.get_page_by_id(page_id, expand="body.storage", status=None, version=None)['body']['storage']['value']
    # Parse the HTML content
    soup = BeautifulSoup(html_content, "html.parser")

    # Extract text from the HTML content
    text = soup.get_text()

    # Find and store attachments
    attachments = []
    attachments_elements = soup.find_all("ri:attachment", {"ri:filename": True})
    for attachment_element in attachments_elements:
        attachment_name = attachment_element["ri:filename"]
        attachments.append(attachment_name)

    return attachments, text

def get_chunks(text):
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=100,
      chunk_overlap=20
      )
  chunks = text_splitter.split_text(text)
  return chunks
